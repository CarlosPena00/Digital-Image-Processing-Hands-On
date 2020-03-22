import cv2
import numpy as np
from argparse import ArgumentParser
from multiprocessing import Pool

LUT_noise, LUT_dst = {}, {}
MAX_P, MAX_M       = 5, 3
title_window       = 'chapter05-adaptive_median_filter'
p_label            = 'Noise: \n10 * P %'
m_label            = 'M Max'

def arg_parser():
    """Arg parser"""    
    parser = ArgumentParser()
    parser.add_argument('-i','--input', help='input image')
    
    return parser
    
def get_salt_pepper_noise(src, P, use_salt=True, use_pepper=True):
    noise_src = src.copy()
    noise     = np.random.rand(*src.shape)
    p_p       = noise < P/2
    p_s       = np.all(((P > noise), (noise  > P/2)),axis=0)
    
    if use_salt:
        noise_src[p_p]  = 0
    if use_pepper:
        noise_src[p_s]  = 255
    return noise_src

def get_neighborhood(src, x, y, m):
    neigh = src[y-m: y+m+1, x-m:x+m+1] 
    return neigh

def get_min_med_max(neigh):
    min_val  = np.min(neigh)
    med_val  = np.median(neigh)
    max_val  = np.max(neigh)
    
    return min_val, med_val, max_val


def adaptive_median_filter(noise_src, M_MAX):
    noise_src_wb      = cv2.copyMakeBorder(noise_src, M_MAX, M_MAX, M_MAX, M_MAX, borderType=cv2.BORDER_REFLECT).astype(np.int)
    dst               = np.zeros_like(noise_src_wb, dtype=np.int)
    height, width, *_ = dst.shape

    for col in (range(M_MAX, width-M_MAX)): # Padding to just apply filter over the original pixels
        for row in range(M_MAX, height-M_MAX):
            dst[row, col] = stage_A(noise_src_wb, col, row, M_MAX)
        
    dst = dst[M_MAX:height-M_MAX, M_MAX:width-M_MAX].astype(np.uint8) # crop to the original size
    return dst

def stage_A(src, x, y, M_MAX, m=1):
    neigh             = get_neighborhood(src, x, y, m)
    zmin, zmed, zmax  = get_min_med_max(neigh)
    
    A1 = zmed - zmin
    A2 = zmed - zmax
    
    if A1 > 0 and A2 < 0:
        return stage_B(src[y, x], zmin, zmed, zmax)
    if m < M_MAX:
        return stage_A(src, x, y, M_MAX, m+1)
    else:
        return zmed

def stage_B(zcurr, zmin, zmed, zmax):

    B1     = zcurr - zmin
    B2     = zcurr - zmax
    
    if B1 > 0 and B2 < 0:
        return zcurr
    return zmed
    

def update_win():
    P          = cv2.getTrackbarPos(p_label, title_window)
    M_MAX      = max(cv2.getTrackbarPos(m_label, title_window), 1)
    noise_src  = LUT_noise[P]
    dst        = LUT_dst[P][M_MAX]
    dst        = np.concatenate((src, noise_src, dst), axis=1).astype(np.uint8)
    
    cv2.imshow(title_window, dst)
    
def generate_noise(src, P):
    
    noise_src = get_salt_pepper_noise(src, P/10)
    mse_noise = np.mean((src.astype(np.int) - noise_src) ** 2)
    mse_text  = f"{mse_noise:0.2f}"
    
    return noise_src#, mse_text

def generate_lut_p(P):
    global LUT_noise
    
    LUT_P_dst = []
    for M in range(0, MAX_M + 1):
        
        if M == 0: 
            LUT_P_dst.append([])
            continue
               
        dst       = adaptive_median_filter(LUT_noise[P], M)
        mse_dst   = np.mean((src.astype(np.int) - dst) ** 2)
        mse_text  = f"{mse_dst:0.2f}"
        
        (label_width, label_height), baseline = cv2.getTextSize(mse_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
        
        cv2.putText(dst, mse_text, ((dst.shape[1] - label_width//2 )//2, 
                    dst.shape[0] - baseline), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    
        LUT_P_dst.append(dst)
        
    return LUT_P_dst, P
        
import time

if __name__ == "__main__":
    parser       = arg_parser()
    args         = parser.parse_args()
    src_path     = args.input
    src          = cv2.imread(src_path, 0)
    flags        = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO 

    if src is None:
        print(f"Input image path not found: {src_path}")
        exit(0)  
        
    
    for P in range(0, MAX_P + 1):
        noise        = generate_noise(src, P)
        LUT_noise[P] = noise
    
    print("Warring: this code could take some time to load")

    pool = Pool(processes=4) 
    for ret in pool.imap_unordered(generate_lut_p, (range(0, MAX_P + 1)) ):
        LUT_P_dst, P = ret
        LUT_dst[P] = LUT_P_dst 
    pool.close()
        
    cv2.namedWindow(title_window, flags)
    cv2.createTrackbar(p_label, title_window , 0, MAX_P, lambda _ : _)
    cv2.createTrackbar(m_label, title_window , 1, 3, lambda _ : _)
    
    while(cv2.waitKey(30) == -1):
        
        update_win()
    
    cv2.destroyAllWindows()
    