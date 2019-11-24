import time, sys, os, copy
import numpy as np
import matplotlib.pyplot as plt

from itertools import product, combinations
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import juggle_axes
from mpl_toolkits.mplot3d import Axes3D, proj3d
import matplotlib.gridspec as gridspec
from matplotlib import animation

def pc_fun_weights(f_vec, weights):
    if weights.size != f_vec.size or np.fabs(np.sum(weights) - 1)>1e-5:
        raise Exception("Dimension mismatch!")
    numBin = weights.size
    xtmp = np.cumsum(weights)
    xx = [ [xtmp[i]]*2 for i in range(0, numBin-1) ]  
    xx = [0]+list(np.array(xx).reshape(-1)) + [1.0]
    yy = [ [f_vec[i], f_vec[i+1]] for i in range(0, numBin-1) ] 
    yy = [f_vec[0]] + list(np.array(yy).reshape(-1)) + [f_vec[-1]]
    return xx, yy

class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

def anim3dplots(X_list, Y_list, Z_list, weights_list = None, info_list = None, radius = 1,\
                     INCLUDE_FUN = True, INCLUDE_WEIGHT = True, FILE_NAME = "", ADD_TIME = True, interval = 1000):
    
    '''When numNeuro = 3, plot points in a cube.'''
    ''' the 'radius' argument acts the same way as maxium FP.'''
        
    if len(X_list) != len(Y_list) or len(Y_list)!= len(Z_list):
        raise Exception('Wrong dimension of inputs!')
    
    elif len(X_list[0])!=len(Y_list[0]) or len(Y_list[0])!=len(Z_list[0]):
        raise Exception('Wrong dimension of inputs!')
        
    num_frames = len(X_list) # number of frames
    num_pts = len(X_list[0]) # number of points
    
    if weights_list is None:
        PLOT_WEIGHTS = False
    elif len(weights_list) != num_frames or len(weights_list[0])!= num_pts:
        raise Exception('Wrong dimension of inputs: weight_list')
    elif np.fabs(np.sum(weights_list[0]) - 1)>1e-5:
        raise Exception('Wrong input of weights: sum error!')
    else:
        PLOT_WEIGHTS = True
    
    # figure setup
    if INCLUDE_FUN:
        fig = plt.figure(figsize = (12,6))

        gs0 = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
        if INCLUDE_WEIGHT:
            gs01 = gridspec.GridSpecFromSubplotSpec(4,1,subplot_spec=gs0[1])
        else:
            gs01 = gridspec.GridSpecFromSubplotSpec(4,1,subplot_spec=gs0[1])

        ax_cube = fig.add_subplot(gs0[0], projection='3d')
        ax_cube.set_aspect("equal")
        ax_f1 = fig.add_subplot(gs01[0])
        ax_f2 = fig.add_subplot(gs01[1])
        ax_f3 = fig.add_subplot(gs01[2])
        if INCLUDE_WEIGHT:
            ax_w = fig.add_subplot(gs01[3])
            weight_max = np.max(np.array(weights_list))
            ax_w.set_ylim(0, weight_max + 0.05)
    else:
        fig = plt.figure(figsize = (8,8))
        ax_cube = fig.gca(projection='3d')
        ax_cube.set_aspect("equal")
        

    # draw cube
    r = [0, radius]
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s-e)) == r[1]-r[0]:
            ax_cube.plot3D(*zip(s, e), color="k", alpha = 0.5, lw = 1.5)#linestyle='--'

    r2 = [0, 1.4*radius]
    for s in np.array(list(product(r2, r2, r2))):
        if np.sum(s> 0) == 1:
            a = Arrow3D(*zip([0,0,0], s), mutation_scale=20,
                lw=1, arrowstyle="-|>", color="k")
            ax_cube.add_artist(a)
    ax_cube._axis3don = False
    ax_cube.grid(True)
    ax_cube.set_xlim([0,1.5*radius])
    ax_cube.set_ylim([0,1.5*radius])
    ax_cube.set_zlim([0,1.5*radius])
    endpts = [[1.45*radius,-0.1*radius,0],[0.1*radius,1.45*radius,0],[0,0,1.45*radius]]
    
    
    for k in range(len(endpts)):
        ax_cube.text(endpts[k][0],endpts[k][1],endpts[k][2],r'$f_%d$'%(k+1),color = 'k',fontsize = 16)


    ax_cube.view_init(azim = 30)

    # color_arr = np.arange(num_pts)
    cmap = plt.cm.get_cmap('nipy_spectral', num_pts) # other colormap names: cubehelix,viridis,...
    color_arr = np.array([cmap(i) for i in range(num_pts)])
    
    scat = ax_cube.scatter(X_list[0], Y_list[0], Z_list[0],  c = color_arr,s= 100, edgecolor='k')
    if info_list is not None:
        x0, y0, _ = proj3d.proj_transform(0,-0.3*radius, -1.5*radius,ax_cube.get_proj())
        info_txt = ax_cube.text2D(x0,y0,"", fontsize = 15)
    
    
    txt_list = [ax_cube.text2D(0,0,"",fontsize = 14, color = 'k') for _ in range(num_pts)]  # 'steelblue'
    if INCLUDE_FUN:
        line1, = ax_f1.plot([], [], color = 'crimson', lw = 2)
        line2, = ax_f2.plot([], [], color = 'seagreen', lw = 2)
        line3, = ax_f3.plot([], [], color = 'steelblue', lw = 2)
        ax_f_list = [ax_f1, ax_f2, ax_f3]
        for ax_f in ax_f_list:
            ax_f.set_xlim([0,1])
            ax_f.set_ylim([0,radius*1.01])
            ax_f.set_yticks([0,radius])
            ax_f.spines['top'].set_visible(False)
            ax_f.spines['right'].set_visible(False)
            ax_f.yaxis.set_ticks_position('left')
            ax_f.xaxis.set_ticks_position('bottom')
            #ax_f.arrow(0,0,1,0,fc='k', ec='k', lw =0.1,head_width=0.05, head_length=0.02,overhang = 0.1,\
            #           length_includes_head= False, clip_on = False)
            #ax_f.arrow(0,0,0,radius,fc='k', ec='k', lw =0.1,head_width=0.01*radius, head_length=0.1*radius,\
            #           overhang = 0.1*radius,\
            #           length_includes_head= False, clip_on = False)
        if INCLUDE_WEIGHT:
            barcollection = ax_w.bar(np.arange(num_pts),weights_list[0], color = color_arr)
    #     plt.tight_layout()#pad = 1.5
    #     gs0.tight_layout(fig)
    #     def init():     
    #         return scat,

    # animation function.  This is called sequentially
    def animate(i):

    #         scat.set_array(color_arr)
        scat._offsets3d = juggle_axes(X_list[i], Y_list[i], Z_list[i],'z') #'z',color=color_set[0],s= 100
        scat.set_sizes(np.ones(num_pts)*100)
        
        for txt, new_x, new_y, new_z, weight in zip(txt_list, X_list[i], Y_list[i], Z_list[i],weights_list[i]):
            # animating Text in 3D proved to be tricky. Tip of the hat to @ImportanceOfBeingErnest
            # for this answer https://stackoverflow.com/a/51579878/1356000
            x_, y_, _ = proj3d.proj_transform(new_x+0.1*radius, new_y+0.1*radius, new_z+0.1*radius, \
                                              ax_cube.get_proj())
            txt.set_position((x_,y_))
            txt.set_text('%.2f'%weight)
        if INCLUDE_FUN:
            x1, y1 = pc_fun_weights(X_list[i],weights_list[i])
            x2, y2 = pc_fun_weights(Y_list[i],weights_list[i])
            x3, y3 = pc_fun_weights(Z_list[i],weights_list[i])
            line1.set_data(x1, y1)
            line2.set_data(x2, y2)
            line3.set_data(x3, y3)

        if INCLUDE_WEIGHT:
            for j, b in enumerate(barcollection):
                b.set_height(weights_list[i][j])
            
        if info_list is not None:
            info_txt.set_text("MI = %.4f"%info_list[i])
        return scat

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate,
                                   frames = num_frames, interval=interval)#, blit=True      
    if ADD_TIME: 
        timestr = time.strftime("%m%d-%H%M%S")
    else:
        timestr = ""
    filename =  FILE_NAME + timestr + ".mp4" 
    directory = os.path.dirname(filename)
    if directory != "":
        try:
            os.stat(directory)
        except:
            os.makedirs(directory)  
    anim.save(filename, writer="ffmpeg")
