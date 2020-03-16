import cv2
import numpy as np
from argparse import ArgumentParser

def arg_parser():
    """Arg parser"""    
    parser = ArgumentParser()
    parser.add_argument('-i','--input', help='input image')
    
    return parser
    
def get_n_noise_imgs(src, n_images=10, mean=0, std=10):
    """ Return a list of images with Gaussian noise. """

    noise_imgs = []
    for idx in range(n_images):
        random_noise = np.random.normal(mean, std, src.shape).round()
        src_n        = (src + random_noise).clip(0, 255).astype(np.uint8)
        noise_imgs.append(src_n)
        
    return noise_imgs


def update_win(src, N, mean, std):
    noise_imgs = get_n_noise_imgs(src, N, mean=mean, std=std)
    mean_img   = (np.sum(noise_imgs,axis=0)/len(noise_imgs)).round().astype(np.uint8)
    mse_mean   = np.mean((src - mean_img) ** 2)
    mse_text   = f"{mse_mean:0.2f}"
    dst        = np.concatenate((src,  mean_img), axis=1)
    
    (label_width, label_height), baseline = cv2.getTextSize(mse_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
    cv2.putText(dst, mse_text, ((dst.shape[1] - label_width//2 )//2, dst.shape[0] - baseline), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    
    cv2.imshow(title_window, dst)
    
    
if __name__ == "__main__":
    parser       = arg_parser()
    args         = parser.parse_args()
    src_path     = args.input
    src          = cv2.imread(src_path)
    title_window = 'chapter02-noise_removal'
    flags        = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO 
    cv2.namedWindow(title_window, flags)
    
    if src is None:
        print(f"Input image path not found: {src_path}")
        exit(0)
        
    cv2.createTrackbar('N images', title_window , 1, 100, lambda _: _)
    cv2.createTrackbar('Std', title_window , 10, 150, lambda _: _)
    cv2.createTrackbar('Mean', title_window , 0, 150, lambda _: _)
    
    while(cv2.waitKey(100) == -1):
        N    = max(cv2.getTrackbarPos('N images', title_window),1)
        mean = cv2.getTrackbarPos('Mean', title_window)
        std  = cv2.getTrackbarPos('Std', title_window)
        update_win(src, N, mean, std)
    
    cv2.destroyAllWindows()
    