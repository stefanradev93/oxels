
import numpy as np
import cv2

class ImagePerspectiveTransform:
    def __init__(self,
                 w,
                 h,
                 frac_keep=0.125,
                 max_shift=None,
                 std_matrix_noise=0.1):
        
        self.w = int(w)
        self.h = int(h)
        self.frac_keep = frac_keep

        if max_shift is None:
            max_shift = h//4

        self.max_shift = max_shift
        self.std_matrix_noise = std_matrix_noise

        self.X, self.Y = np.meshgrid(np.arange(w), np.arange(h))

    def get_index_permutation(self, H, sub_x, sub_y, mask=None):
        H_inv = np.linalg.inv(H)
        l = 1/(H_inv[2,0]*self.X + H_inv[2,1]*self.Y + H_inv[2,2])
        X_ = (-H_inv[0,0]*H[0,2] - H_inv[0,1]*H[1,2] + H_inv[0,0]*self.X + H_inv[0,1]*self.Y)*l
        Y_ = (-H_inv[1,0]*H[0,2] - H_inv[1,1]*H[1,2] + H_inv[1,0]*self.X + H_inv[1,1]*self.Y)*l


        ind = np.round(np.stack([Y_.ravel(), X_.ravel()], axis=1)).astype(int)

        match = ((ind[:,1] > sub_x)*(ind[:,1] < self.w+sub_x)*(ind[:,0] > sub_y)*(ind[:,0] < self.h+sub_y)).reshape((-1,1))
        match *= np.random.random(match.shape) < self.frac_keep

        random_ind = np.stack([np.random.randint(0,self.h,self.w*self.h), np.random.randint(0,self.w,self.w*self.h)], axis=1)
        ind = match*(ind - [[sub_y, sub_x]]) + (1 -  match)*random_ind

        if not mask is None:
            match *= ((mask == 0).ravel()[ind[:,0]*self.w + ind[:,1]]).reshape(match.shape)
            ind = match*ind + (1 -  match)*random_ind


        ind_flat = ind[:,0]*self.w + ind[:,1]
        return ind_flat, match.ravel()
    
    def get_views_and_permutation(self, im):
        if im.shape[0] <= self.h+2*self.max_shift or im.shape[1] <= self.w+2*self.max_shift:
            im = cv2.resize(im, (max(im.shape[1], self.w+2*self.max_shift+1), max(im.shape[0], self.h+2*self.max_shift+1)))
            
        x = np.random.randint(self.max_shift+self.w//2, im.shape[1]-self.max_shift-self.w//2+1)
        y = np.random.randint(self.max_shift+self.h//2, im.shape[0]-self.max_shift-self.h//2+1)

        alpha = np.random.random()*2*np.pi
        dx,dy = np.random.randint(-self.max_shift, self.max_shift + 1, 2)

        B = np.array([[np.cos(alpha),-np.sin(alpha)], [np.sin(alpha),np.cos(alpha)]])
        B += np.random.normal(0,self.std_matrix_noise,(2,2))

        A = np.eye(3, dtype=np.float32)
        A[:2,:2] = B
        A[:2,2] = -A[:2,:2].dot([self.w//2, self.h//2]) + [self.w//2, self.h//2] #this is relative to view 1 and 2
        
        permutation, flags = self.get_index_permutation(A, dx, dy)

        A[:2,2] = -A[:2,:2].dot([x,y]) + [self.w//2, self.h//2] #this is relative to original image

        view1 = cv2.warpPerspective(im, A, (self.w, self.h), cv2.INTER_AREA)
        view2 = im[y+dy-self.h//2:y+dy+self.h//2, x+dx-self.w//2:x+dx+self.w//2]
        
        valid = np.ones(im.shape[:2])
        mask1 = cv2.warpPerspective(valid, A, (self.w, self.h), cv2.INTER_AREA).ravel() == 1
        mask2 = np.ones(self.h*self.w, dtype=bool)

        return view1, view2, permutation, flags, mask1, mask2
