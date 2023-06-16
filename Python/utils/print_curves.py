import matplotlib.pyplot as plt

def print_pr_curve(precision, recall, title, pr_auc, no_skill, display = False):
    #create precision recall curve
    _, ax = plt.subplots()
    ax.plot(recall, precision, color='purple', label='AUC = %0.4f' % pr_auc)
    ax.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')

    #add axis labels to plot
    ax.set_title(title)
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')

    plt.legend(loc='best')
    if not display:
        #display plot
        plt.show
    else:
        #save plot in a file
        plt.savefig("PR_" + title + '.png')


def print_roc_curve(fpr, tpr, title, roc_auc, display = False):
    #create ROC curve
    _, ax = plt.subplots()
    ax.plot(fpr, tpr, color='purple', label='AUC = %0.4f' % roc_auc)
    ax.plot([0, 1], [0, 1], linestyle='--', label='No Skill')

    #add axis labels to plot
    ax.set_title(title)
    ax.set_ylabel('True Positive Rate(TPR)')
    ax.set_xlabel('False Positive Rate(FPR)')

    plt.legend(loc='best')
    if not display:
        #display plot
        plt.show
    else:
        #save plot in a file
        plt.savefig("ROC_" + title + '.png')