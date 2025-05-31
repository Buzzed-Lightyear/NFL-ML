import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

def plot_confusion_matrix_heatmap(y_test, y_pred, model_name: str, ax=None):
    """
    Plot confusion matrix heatmap.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        model_name: Name of the model for plot title 
        If ax is None, a new figure and axes will be created.
    """
    show_plot = False
    if ax is None:
        # This allows the function to be used standalone if needed
        fig, ax = plt.subplots(figsize=(8, 6))
        show_plot = True

    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted 0 (Loss/Tie)', 'Predicted 1 (Win)'],
                yticklabels=['Actual 0 (Loss/Tie)', 'Actual 1 (Win)'],
                ax=ax) # Use the provided ax
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(f'Confusion Matrix for {model_name} Model')

    if show_plot: 
        plt.show() 

def plot_roc_auc_curve(y_test, y_pred_proba, model_name: str, ax=None):
    """
    Plot ROC curve.
    
    Args:
        y_test: True labels
        y_pred_proba: Predicted probabilities
        model_name: Name of the model for plot title
    """
    show_plot = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6)) # Create fig and ax
        show_plot = True
        
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})') # Use ax.plot
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Use ax.plot
    ax.set_xlim([0.0, 1.0]) # Use ax.set_xlim
    ax.set_ylim([0.0, 1.05]) # Use ax.set_ylim
    ax.set_xlabel('False Positive Rate') # Use ax.set_xlabel
    ax.set_ylabel('True Positive Rate') # Use ax.set_ylabel
    ax.set_title(f'ROC Curve for {model_name} Model') # Use ax.set_title
    ax.legend(loc="lower right") # Use ax.legend
    
    if show_plot: 
        plt.show()

def plot_model_feature_importances(model, feature_names: list, model_name: str, ax=None):
    show_plot = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 12)) # Create fig and ax
        show_plot = True

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Ensure coef_ is 1D for this visualization, or select the relevant part
        coef_data = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
        importances = np.abs(coef_data)

    else:
        if show_plot: # If we created a figure but there's nothing to plot
             plt.close(fig if 'fig' in locals() else plt.gcf()) # Close the empty figure
        print(f"Model {model_name} does not have feature_importances_ or coef_ attribute.")
        return # Exit if no importances to plot
    
    if len(feature_names) != len(importances):
        if show_plot:
             plt.close(fig if 'fig' in locals() else plt.gcf())
        print(f"Warning: Number of feature names ({len(feature_names)}) does not match number of importances ({len(importances)}) for model {model_name}.")
        # Decide how to handle: return, or plot with generic feature names, etc.
        # For now, let's return to avoid errors if this happens.
        return


    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    
    print(f"\nTop 10 most important features ({model_name}):")
    print(feature_importance_df.head(10))
    
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(30), ax=ax) # Use ax
    ax.set_title(f'Feature Importances from {model_name} Model (Top 30)') # Use ax.set_title
    
    # tight_layout might need to be called on the figure if you have access to it
    # For simplicity here, we assume it's handled by the caller if 'ax' is passed,
    # or plt.tight_layout() if 'ax' was None and fig was created.
    if 'fig' in locals() and fig is not None: # if fig was created in this function
        fig.tight_layout()
    elif ax.figure is not None: # if ax was passed, use its figure
        ax.figure.tight_layout()


    if show_plot:
        plt.show()