
from sklearn.metrics import precision_score,accuracy_score,recall_score,f1_score


def custom(label,predict,average='macro'):
    acc=accuracy_score(label,predict)
    pre=precision_score(label,predict, average=average)
    rec=recall_score(label,predict, average=average)
    f1=f1_score(label,predict, average=average)
    return acc,pre,rec,f1