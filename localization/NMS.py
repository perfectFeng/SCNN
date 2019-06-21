import numpy as np


    
def non_maxima_suppression(dets, score , cls, vid, overlap = 0.7, measure = 'iou'):
    '''
    非极大值抑制

    dets:ndaray
        [num_segments,2],each row [f_init, f_end]
    score:ndarray
        [num_segment]
    '''
    
    measure = measure.lower()
    if score is None:
        score = dets[:,1]
    if score.shape[0] != dets.shape[0]:
        raise ValueError('mismatch between deys and score')
    if dets.dtype.kind == 'i':
        dets = dets.astype('float')

    #discard incorrect segment
    #选出f_end>f_init的segment的id
    idx_correct = np.where(dets[:, 1] > dets[:, 0])[0]

    t1 = dets[idx_correct, 0]
    t2 = dets[idx_correct, 1]
    area = t2 - t1 + 1

    idx = np.argsort(score[idx_correct])
    pick = []
    #先挑选出
    while len(idx) > 0:
        last = len(idx) - 1
        i = idx[last]
        pick.append(i)

        tt1 = np.maximum(t1[i], t1[idx])
        tt2 = np.maximum(t2[i], t2[idx])

        wh = np.maximum(0, tt2-tt1+1)
        if measure == 'overlap':
            o = wh / area[idx]
        elif measure == 'iou':
            o = wh / (area[i] + area[idx] - wh)
        else:
            raise ValueError('Unkown overlap measure for NMS')

        idx = np.delete(idx, np.where(o>overlap)[0])

    return dets[idx_correct[pick], :].astype('int'), cls[idx_correct[pick]], score[idx_correct[pick]],vid[idx_correct[pick]]

           
file = open('D:/冯咩咩的文件夹/soccer_detection/SCNN/localization/prepredict/final_result/final_result/corner/0011.txt')
l =list(file)
lines = []
vid = []
loc = []
cls = []
score = []


    
