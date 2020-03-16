import cv2
import numpy as np
from argparse import ArgumentParser

def arg_parser():
    """Arg parser"""    
    parser = ArgumentParser()
    parser.add_argument('-i','--input', help='input image')
    
    return parser
    
def get_segmented_img(src, num_colors=2):
    """ Return the src image with just num_colors. """
    vector             = src.reshape(-1).astype(np.float32) # 3 channel to 1 
    criteria           = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 1.0)
    _, labels, centers = cv2.kmeans(vector, num_colors, None, criteria, attempts=5, flags=cv2.KMEANS_PP_CENTERS)
    # center: value of kmeans cluster center, in other words the final color 
    # label: index of associeted cluster per pixel

    labels_w_color  = centers[labels.flatten()] 
    segmented       = labels_w_color.reshape((src.shape)).astype(np.uint8)
    return segmented 


def update_win(src, N):
    seg_img    = get_segmented_img(src, N)
    mse_mean   = np.mean((src - seg_img) ** 2)
    mse_text   = f"{mse_mean:0.2f}"
    dst        = np.concatenate((src,  seg_img), axis=1)

    (label_width, label_height), baseline = cv2.getTextSize(mse_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
    cv2.putText(dst, mse_text, ((dst.shape[1] - label_width//2 )//2, dst.shape[0] - baseline), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow(title_window, dst)
    
    
if __name__ == "__main__":
    parser       = arg_parser()
    args         = parser.parse_args()
    src_path     = args.input
    src          = cv2.imread(src_path, 0)
    title_window = 'chapter02-noise_removal'
    flags        = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO 
    cv2.namedWindow(title_window, flags)
    
    if src is None:
        print(f"Input image path not found: {src_path}")
        exit(0)
        
    cv2.createTrackbar('N', title_window , 2, 100, lambda _: _)
    
    while(cv2.waitKey(100) == -1):
        N    = max(cv2.getTrackbarPos('N', title_window),1)
        update_win(src, N)
    
    cv2.destroyAllWindows()
    