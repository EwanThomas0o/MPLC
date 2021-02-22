import matplotlib.pyplot as plt
import numpy as np
import cmath as cm
import matplotlib.animation as ani 
from IPython.display import display, clear_output, HTML 
import imageio

def saveAndDisplay(animation, filename, fps):
    Writer = ani.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)

    animation.save(filename, writer = writer, dpi = 200)
    
    display(HTML(animation.to_jshtml()))
        
def AngSpec(image, intervals, wavelength, Min, Max ,sqT):
    intervals += 1
    animationValues, incrementValues = getAnimationValues(intervals, Min, Max, wavelength, image, sqT)
    animation = createAnimation(intervals, animationValues, incrementValues)
    return animation

def getAnimationValues(intervals, Min, Max, wavelength, image, sqT):
    img = imageio.imread(image)[:, :, 0]
    
    incrementValues = np.linspace(Min, Max, intervals)
    animationValues = np.empty(img.shape)
    
    for i in range(intervals):
        z = incrementValues[i]
        frameValues = propagate(image, wavelength, z, sqT)
        animationValues = np.dstack([animationValues, frameValues])
        
    return animationValues, incrementValues

def createAnimation(intervals, animationValues, incrementValues):
    fig, ax = plt.subplots()
    
    def animate(i):
        ax.clear()
        ax.imshow(animationValues[:, :, i + 1], cmap = "gist_gray")
        ax.set_title("z = {}".format(incrementValues[i]))
        
    animation = ani.FuncAnimation(fig, animate, intervals, interval = 200)
    
    return animation

def propagate(image, wavelength, z, sqT):
    xBeam0 = imageio.imread(image)[:, :, 0]
    xBeam0 = np.asarray(xBeam0).astype(complex)
    
    k = 2 * np.pi / wavelength
    
    kBeam = np.fft.fft2(xBeam0)
    
    kx = np.fft.fftfreq(kBeam.shape[0], wavelength / (2 * np.pi))
    ky = np.fft.fftfreq(kBeam.shape[1], wavelength / (2 * np.pi))

    for i in range(kBeam.shape[0]):
        for j in range(kBeam.shape[1]):
            kz = np.sqrt(k*k - kx[i]*kx[i] - ky[j]*ky[j])
            kBeam[i][j] = kBeam[i][j] * cm.rect(1, -1 * z * kz)
    
    xBeam1 = np.fft.ifft2(kBeam)
    
    intensity1 = xBeam1 * np.conj(xBeam1)
    intensity1 = intensity1.real.astype("float64")

    if sqT == True:
        #creating square flat top intensity profile
        _max = np.amax(intensity1)
        intensity1 = np.square(intensity1)
        tolerance = 0.8 * _max
            
        for i in range(intensity1.shape[0]):
            for j in range(intensity1.shape[1]):
                if intensity1[i][j] > tolerance:
                    intensity1[i][j] = tolerance
        
    return intensity1

animation = AngSpec("dog.png", 40, 10, 0, 2e3, False)
saveAndDisplay(animation, "gauss.mp4", 15)