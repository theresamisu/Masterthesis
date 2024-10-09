import pytorch_lightning as pl
from torchmetrics import ConfusionMatrix, JaccardIndex, MetricCollection
from torchmetrics.classification import MulticlassAccuracy
import torch
from model_utils import get_bands
import time 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from eoekoland_dataset import CT_CLASSES_condensed
from torch.optim.lr_scheduler import ReduceLROnPlateau

import visualization_utils

class ModelHandler(pl.LightningModule):

    def __init__(self,
                 model,
                 modalities, 
                 lr: float = 1e-3,
                 classes: list = CT_CLASSES_condensed,
                 epochs: int = None,
                 class_weights: list = [],
                 loss: str = "CE-W") -> None:
        super().__init__()
        
        self.model = model
        self.classes = classes
        self.num_classes = len(self.classes)
        self.lr = lr
        self.epochs = epochs
        self.modalities = modalities
        self.start_time = time.monotonic()
        if len(class_weights)==self.num_classes and loss=="CE-W":
            print("using class weights")
            self.criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights).float())
        else:
            print("unweighted cross entropy")
            self.criterion = torch.nn.CrossEntropyLoss()
        self.plot_every_x = 500 # plots segmentation map of first sample in batch every x batches
        # metrics
        metrics = MetricCollection({
            # accuracy over all samples (independent of class)
            "overall_accuracy": MulticlassAccuracy(num_classes=self.num_classes, average="micro"), 
            # for each class separately and then take mean
            "mean_accuracy": MulticlassAccuracy(num_classes=self.num_classes, average="macro"), 
            "mean_iou": JaccardIndex(task="multiclass", num_classes=self.num_classes, average="macro"),
            "overall_accuracy_nb": MulticlassAccuracy(num_classes=self.num_classes, average="micro", ignore_index=0), 
            # for each class separately and then take mean
            "mean_accuracy_nb": MulticlassAccuracy(num_classes=self.num_classes, average="macro", ignore_index=0), 
            "mean_iou_nb": JaccardIndex(task="multiclass", num_classes=self.num_classes, average="macro", ignore_index=0) 
        }, compute_groups=False) 

        self.train_metrics = metrics.clone(prefix = 'train/')
        self.val_metrics = metrics.clone(prefix = 'val/')
        self.test_metrics = metrics.clone(prefix = 'test/')
        self.pred_metrics = metrics.clone(prefix= 'best_val/')
    
    def forward(self, x):
        return self.model(x)


    def training_step(self, batch, batch_idx):
        inputs, targets = get_bands(batch, self.modalities) # early fusion is done here
        batch_size = len(batch)
        if np.all(targets.detach().cpu().numpy().astype(int)==0):
            print("only background class in patch")
        pred = self(inputs)
        
        self.log('step', self.trainer.current_epoch)
        # update metrics
        self.train_metrics.update(pred, targets)
        
        # compute max class
        out_max = torch.argmax(pred, dim=1)

        # save example to disk
        if batch_idx%self.plot_every_x==0:
            visualization_utils.visualize_example_with_prediction(input_ts=inputs[0], ground_truth=targets[0], predicted_segmentation=out_max[0], batch_idx=batch_idx, epoch=self.epochs, subset='train',name=self.modalities+"_"+self.model.name,modalities=self.modalities, num_classes=self.num_classes)

        # calculate loss
        loss = self.criterion(pred, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, logger=True, batch_size=batch_size)
        
        return {"loss": loss, "pred": pred.detach(), "targets": targets.detach()}

    def training_epoch_end(self, outputs) -> None:
        output = self.train_metrics.compute()
        output["train/mean_accuracy_nb"] = (output["train/mean_accuracy_nb"] * self.num_classes) / (self.num_classes-1)
        
        self.log_dict(output, logger=True, on_step=False, on_epoch=True)
        self.train_metrics.reset()
        
    def validation_step(self, batch, batch_idx):
        inputs, targets = get_bands(batch, modalities=self.modalities)
        batch_size = len(batch)
        pred = self(inputs)
        
        # update metrics
        self.val_metrics.update(pred, targets)
       
       # compute max class
        out_max = torch.argmax(pred, dim=1)

        # save example
        if batch_idx%self.plot_every_x==0:
            visualization_utils.visualize_example_with_prediction(input_ts=inputs[0], ground_truth=targets[0], predicted_segmentation=out_max[0],batch_idx=batch_idx, epoch=self.epochs, subset='val',name=self.modalities+"_"+self.model.name,modalities=self.modalities, num_classes=self.num_classes)

        # calculate loss
        loss = self.criterion(pred, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size) 
        
        return {"loss": loss, "pred": pred.detach(), "targets": targets.detach()}
    
    # accuracy * num_classes/(num_classes-1)
    def validation_epoch_end(self, outputs):
        output = self.val_metrics.compute()
        # correct the mean accuracy that ignores background pixels
        output["val/mean_accuracy_nb"] = (output["val/mean_accuracy_nb"] * self.num_classes) / (self.num_classes-1)
        # try to log every epoch instead of every n steps as specified in the trainer by log_ever_n_steps???
        self.log('step', self.trainer.current_epoch)
        
        self.log_dict(output, logger=True, on_step=False, on_epoch=True)
        self.val_metrics.reset()
        
        outs = torch.cat([tmp['pred'] for tmp in outputs])
        labels = torch.cat([tmp['targets'] for tmp in outputs])

        # calculate confusion matrix and store as figure
        fig_confusion_percent,_ = self.make_confusion_matrix(outs, labels, title=f"Validation Confusion Matrix {self.model.name} epoch={self.current_epoch}", norm="true")
        # with absolute pixel counts
        fig_confusion_abs,_ = self.make_confusion_matrix(outs, labels, title=f"Validation Confusion Matrix {self.model.name} epoch={self.current_epoch}", norm="none")
        
        # add confusion matrix figure to logger
        exp = self.logger.experiment 
        exp.add_figure(tag="val_confusion_matrix_abs", figure=fig_confusion_abs, global_step=self.current_epoch)
        exp.add_figure(tag="val_confusion_matrix_percent", figure=fig_confusion_percent, global_step=self.current_epoch)

        
    def test_step(self, batch, batch_idx):
        inputs, targets = get_bands(batch, self.modalities)
        batch_size = len(batch)
        pred = self(inputs)
        
        # compute max class and show pred. map
        out_max = torch.argmax(pred, dim=1) # BxCxHxW -> BxHxW
        
        if batch_idx%self.plot_every_x==0:
            visualization_utils.visualize_example_with_prediction(input_ts=inputs[0], ground_truth=targets[0], predicted_segmentation=out_max[0],batch_idx=batch_idx, epoch=self.epochs, subset='test', name=self.modalities+"_"+self.model.name, modalities=self.modalities, num_classes=self.num_classes)

        # update metrics
        self.test_metrics.update(pred, targets)
        
        # calculate loss
        loss = self.criterion(pred, targets)
        self.log("loss", loss, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        
        return {"loss": loss, "pred": pred, "targets": targets}

    def test_epoch_end(self, outputs) -> None:
        output = self.test_metrics.compute()
        output["test/mean_accuracy_nb"] = (output["test/mean_accuracy_nb"] * self.num_classes) / (self.num_classes-1)
        
        self.log_dict(output, logger=True) #, batch_size=self.batch_size)
        self.test_metrics.reset()
        
        outs = torch.cat([tmp['pred'] for tmp in outputs])
        labels = torch.cat([tmp['targets'] for tmp in outputs])

        # calculate confusion matrix and store as figure
        fig_confusion_percent, class_accuracy = self.make_confusion_matrix(outs, labels, title=f"Test Confusion Matrix {self.model.name}", norm="true")
        print(class_accuracy)
        # with absolute pixel counts
        fig_confusion_abs,_ = self.make_confusion_matrix(outs, labels, title=f"Test Confusion Matrix {self.model.name}", norm="none")


        exp = self.logger.experiment 
        exp.add_figure(tag="test_confusion_matrix_abs", figure=fig_confusion_abs)  
        exp.add_figure(tag="test_confusion_matrix_percent", figure=fig_confusion_percent)    

    def validate_step(self, batch, batch_idx):
        inputs, targets = get_bands(batch, self.modalities)
        #mask = batch["p_mask"]
        pred = self(inputs)
        
        # compute max class and show pred. map
        out_max = torch.argmax(pred, dim=1)
        # only if not only background class
        if batch_idx%self.plot_every_x==0 and not np.all(targets.detach().cpu().numpy().astype(int)==0):
            visualization_utils.visualize_example_with_prediction(input_ts=inputs[0], ground_truth=targets[0], predicted_segmentation=out_max[0],batch_idx=batch_idx, epoch=self.epochs, subset='best_val', name=self.modalities+"_"+self.model.name, modalities=self.modalities, num_classes=self.num_classes)

        # update metrics
        self.pred_metrics.update(pred, targets)
        
        # calculate loss
        loss = self.criterion(pred, targets)
        
        return {"loss": loss, "pred": pred, "targets": targets}
    
    def validate_epoch_end(self, outputs) -> None:
        output = self.pred_metrics.compute()
        output["best_val/mean_accuracy_nb"] = (output["best_val/mean_accuracy_nb"] * self.num_classes) / (self.num_classes-1)
        
        self.pred_metrics.reset()
        
    def predict_step(self, batch, batch_idx):
        inputs, targets = get_bands(batch, self.modalities)
        pred = self(inputs)
        
        # compute max class and show pred. map
        out_max = torch.argmax(pred, dim=1)
        
        return out_max, targets, pred
    
    
            
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr) # weight_decay=0.01, 
        scheduler = ReduceLROnPlateau(optim, patience=5) # Reduce learning rate if no improvement in val loss is seen for 5 epochs
        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "epoch",
                "monitor": "val/loss",
                "frequency": 1
            },
        }
    
    def make_confusion_matrix(self, outs, labels, title, norm="true"):
        confusion = ConfusionMatrix(num_classes=self.num_classes, normalize=norm, task="multiclass").to(outs.device)
        confusion(outs, labels)
        conf = confusion.compute().detach().cpu().numpy()
        computed_confusion = np.round(conf,3)
        
        axis_labels = [name+", "+str(idx) for (name, idx) in zip(self.classes, np.arange(self.num_classes))]
        df_cm = pd.DataFrame(computed_confusion, index = range(self.num_classes), columns=range(self.num_classes))
        plt.figure(figsize = (11,9))
        ax = sns.heatmap(df_cm, annot=True, cmap='Blues')
        ax.set_xticklabels(axis_labels, rotation=90)
        ax.set_yticklabels(axis_labels, rotation=0)
        fig = ax.get_figure()
        fig.suptitle(title)
        fig.supylabel("Actual")
        fig.supxlabel("Prediction")
        fig.tight_layout()
        plt.close(fig)

        return fig, np.diagonal(conf)