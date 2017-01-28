import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt



def final_issues(test_visual_words_scaled, test_labels, clf, options):
    # Get the predictions:
    predictions = clf.predict(test_visual_words_scaled)

    plot_name = options.file_name + '_test'

    classes = clf.classes_

    # Report file:
    fid = open('report.txt', 'w')
    fid.write(classification_report(test_labels, predictions, target_names=classes))
    fid.close()

    # Confussion matrix:
    compute_and_save_confusion_matrix(test_labels, predictions, options, plot_name)


    #############################################################################


def compute_and_save_confusion_matrix(test_labels, predictions, options, plot_name):
    cnf_matrix = confusion_matrix(test_labels, predictions)
    plt.figure()
    classes = set(test_labels)

    # Prints and plots the confusion matrix.
    plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    print(cnf_matrix)

    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, cnf_matrix[i, j],
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if options.save_plots:
        file_name = 'conf_matrix_' + plot_name + '.png'
        plt.savefig(file_name, bbox_inches='tight')
    if options.show_plots:
        plt.show()


class general_options_class:
    # General options for the system.
    train_data_dir = '../../Databases/MIT/train'
    val_data_dir = '../../Databases/MIT/validation'
    test_data_dir = '../../Databases/MIT/test'
    img_width = 224
    img_height = 224
    number_of_epoch = 20
    batch_size = 32
    val_samples = 807
    # test_samples = 400
    file_name = 'test'
    show_plots = 1
    save_plots = 1
    compute_evaluation = 0  # Compute the ROC, confusion matrix, and write report.
    optimizer = 'adadelta'
    model = 'pool'
    drop_prob_fc = 0.5