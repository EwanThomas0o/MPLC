# Title: MPLC
# Author: Ewan-James Thomas
# filename: MPLC.py
# License: Public Domain




#**************** Multiplane Light Conversion ****************#
#
# Purpose
# -------
#   This program uses mulitplane light conversion to map user defined input modes to user defined output modes via an 
# optimisation algorithm.
# The program allows the user to define a set of input/output electric field modes via image files (or prebuilt functions
# i.e. wave.gaussianBeam()) which are passed through the algorithm in order to find the form of the phase screen that are placed
# inbetween the input and ouput. 
# 
# The Physics
# -----------
#   Between the output and input there are N screens(defined by user) that, once the algorithm is run, have spatially varying phase patterns.
# i.e. The electric field propagates through the screens and free space and picks up a phase exp(i*p_ij) at each pixel where p_ij is the phase value
# of a screen at the ijth pixel. This models the wave passing through screens of varying refractive index (which could be built in the lab). As a 
# result, the electric field eminating from each pixel now interferes with the electric field from every other pixel as such to form the output mode 
# by the time the input mode reaches the end of the array of screens. (See MPLC video link in Week 6 of lab book)
# 
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as F
from math import pi, ceil, radians, sqrt, log, sin, cos, acos, asin, exp
import cmath as c
from scipy import linalg
import matplotlib.animation as ani 
from IPython.display import display, clear_output, HTML
import matplotlib.colors as colors
import imageio
from PIL import Image
import copy
# -----Physical values------
n = 512 #Number of pixels (nxn)
wavelength = 500e-9 # Wavelength in meteres
int_plan_dist = 1e-2 # 1cm between planes
pix_size = 1e-3 #Each pixel is 1mm in size
# --------------------------

# ----- Animation Functions -----#
def SaveAndDisplay(animation, filename, fps):
    Writer = ani.writers['ffmpeg']
    writer = Writer(fps = fps, metadata = dict(artist = 'Me'), bitrate = 1800)

    animation.save(filename, writer = writer, dpi = 300)

    display(HTML(animation.to_jshtml()))

def AngularSpectrum(wave1, wave2, screen_array, intervals, dz):
    animationValues1 = getAnimationValues(wave1, screen_array, intervals, dz, True)
    animationValues2 = getAnimationValues(wave2, screen_array, intervals, dz, False)
    animation = createAnimation(intervals, animationValues1, animationValues2, dz)
    return animation

def getAnimationValues(wave, screen_array, intervals, dz, phase_add):

    animationValues = np.empty(wave.field.shape)
    screen_cache = 0
    for i in range(intervals):
        frameValues = np.abs(wave.field)
        propagate(wave, dz)
        animationValues = np.dstack([animationValues, frameValues])
        if((i+1)%(int_plan_dist/dz)== 0 and i != intervals-1 and i < len(screen_array.screens)*10 and phase_add == True): #times 10 bc 10 frames between each screen
            print("screen {} with z = {}".format(i+1,(i+1)*dz))
            addPhase(wave, phase_array.screens[screen_cache])
            screen_cache += 1

    return animationValues

def createAnimation(intervals, animationValues1, animationValues2, dz):
    fig, (ax1,ax2) = plt.subplots(1,2, sharey=True)

    def animate(i):
        ax1.clear()
        ax1.imshow(animationValues1[:,:,i+1], cmap = 'hot', norm = colors.Normalize(vmin = 0, vmax = 1.0))
        ax1.set_title("With Phase Screens")
        ax1.set(xlabel = "x (mm)", ylabel = "y (mm)")
        ax2.clear()
        ax2.imshow(animationValues2[:,:,i+1], cmap = 'hot', norm = colors.Normalize(vmin = 0, vmax = 1.0))
        ax2.set_title("Without Phase Screens")
        ax2.set(xlabel = "x ($mm)")

        fig.suptitle("z = {:.3f}$cm$".format(0.1*i))
        # TODO: add phase at right time!

    animation = ani.FuncAnimation(fig, animate, intervals, interval = 200)

    return animation    
#---------------#

# Class that defines the Electric wave
class wave():
    def __init__(self, area: np.ndarray, wavelength:float):
        self.area = area # Size of screen
        self.field = np.zeros(area, dtype=np.complex_) # Complex valued Electric field
        self.wavelength = wavelength # Physical Wavlength
        self.mod_k = 2*np.pi/wavelength # K vector magnitude

    # Using equation of a gaussian to create an electric field. Imaginary component at all pixels is zero
    def gaussianBeam(self, amplitude:float, sigma:float):
        for i in range(self.area[0]):
            for j in range(self.area[1]): 
                self.field[i][j] = amplitude * np.exp(-(((i-self.area[0]/2)*pix_size)/np.sqrt(2)*sigma)**2 -(((j-self.area[1]/2)*pix_size)/np.sqrt(2)*sigma)**2)

    def gaussianBeamOffCentreOne(self, amplitude:float, sigma:float):
            for i in range(self.area[0]):
                for j in range(self.area[1]): 
                    self.field[i][j] = amplitude * np.exp(-(((i-self.area[0]/2)*pix_size)/np.sqrt(2)*sigma)**2 -(((j-self.area[1]/3)*pix_size)/np.sqrt(2)*sigma)**2)
    
    def gaussianBeamOffCentreTwo(self, amplitude:float, sigma:float):
            for i in range(self.area[0]):
                for j in range(self.area[1]): 
                    self.field[i][j] = amplitude * np.exp(-(((i-self.area[0]/2)*pix_size)/np.sqrt(2)*sigma)**2 -(((j-self.area[1]*2/3)*pix_size)/np.sqrt(2)*sigma)**2)
    
    #Gaussian beam with pi phase difference between beam on left and beam on right
    def doubleGaussOne(self, amplitude, sigma):
        for i in range(self.area[0]):
            for j in range(self.area[1]):
                self.field[i][j] = amplitude * np.exp(-(((i-self.area[0]/2)*pix_size)/np.sqrt(2)*sigma)**2 -(((j-self.area[1]/3)*pix_size)/np.sqrt(2)*sigma)**2) + np.exp(-(((i-self.area[0]/2)*pix_size)/np.sqrt(2)*sigma)**2 -(((j-self.area[1]*2/3)*pix_size)/np.sqrt(2)*sigma)**2)
                if(j < self.area[1]/2):
                    self.field[i][j] *= c.rect(1, np.pi)
    
    def doubleGaussTwo(self, amplitude, sigma):
        for i in range(self.area[0]):
            for j in range(self.area[1]):
                self.field[i][j] = amplitude * np.exp(-(((i-self.area[0]/2)*pix_size)/np.sqrt(2)*sigma)**2 -(((j-self.area[1]/3)*pix_size)/np.sqrt(2)*sigma)**2) + np.exp(-(((i-self.area[0]/2)*pix_size)/np.sqrt(2)*sigma)**2 -(((j-self.area[1]*2/3)*pix_size)/np.sqrt(2)*sigma)**2)

# Function that propagates the wave through a given distance dz
def propagate(wave, dz):

    kBeam = F.fft2(wave.field)
    kx = F.fftfreq(kBeam.shape[0], 1/wave.mod_k)
    ky = F.fftfreq(kBeam.shape[1], 1/wave.mod_k)

    for i in range(kBeam.shape[0]):
        for j in range(kBeam.shape[1]):
            kz = np.sqrt(wave.mod_k*wave.mod_k -(kx[i]*kx[i]+ky[j]*ky[j]))
            kBeam[i][j] *= c.rect(1, kz*dz)


    newBeam = F.ifft2(kBeam)
    newBeam[0,:] = 0
    newBeam[:,0] = 0
    newBeam[wave.area[0]-1, :] = 0
    newBeam[wave.area[1]-1, :] = 0
    wave.field = newBeam
    return wave

# Spatial pixel level overlap of two waves. returns an area of dim area and is subtracted from old phase elementwise to get new phase
def overlap(field1, field2):
    if field1.area != field2.area:
        return ValueError("Input field and output field must have same area")
    overlap = np.zeros(field1.area, dtype=np.complex_)
    for i in range(field1.area[0]):
        for j in range(field1.area[1]):
            overlap[i][j] = field1.field[i][j] * np.conj(field2.field[i][j])
    return overlap

# Using overlap, finding the phase of a given screen
def newPhase(overlap, screen):
    newPhase = np.zeros(screen.shape, dtype=float)
    for i in range(overlap.shape[0]):
        for j in range(overlap.shape[1]):
            newPhase[i][j] = screen[i][j] - np.angle(overlap[i][j])
    return newPhase

def addPhase(wave, phase_screen):
    if(wave.field.shape == phase_screen.shape):
        for i in range(wave.area[0]):
            for j in range(wave.area[1]):
                wave.field[i][j] *= c.rect(1, phase_screen[i][j]) #adding the phase by x complex exponential
    return wave

def initial_wave_field(image, wave):
    img = Image.open(image).resize((wave.area))
    img = img.convert('L')
    WIDTH, HEIGHT = img.size
    data = list(img.getdata())
    data = [data[offset:offset+WIDTH] for offset in range(0, WIDTH*HEIGHT, WIDTH)]
    data = np.asarray(data, dtype=complex)/255

    return data

# Series of arrays that shape the light so that the input transforms to a desired output
# So far, this function only works for one input mode.
# TODO: Transform a gaussian into itself. i.e. refocus using each plane so we don't loose power. Using monochromatic light.
# TODO: Generalise for n input/output modes
class mulitplane_phase_array():
    def __init__(self, N:int, separation:float, input_modes, output_modes):
        self.N = N
        self.wavelength = input_modes[0].wavelength
        self.separation = separation
        self.screens = []
        self.input_modes = input_modes
        self.output_modes = output_modes
        self.area = input_modes[0].area
        if (len(input_modes) != len(output_modes)):
            return

    def initScreens(self, f:float):
        lens = np.zeros(self.area, dtype=float)
        for i in range(self.area[0]):
            for j in range(self.area[1]):
                # Step 0: Setting all screens to lens that focuses in the fwd direction
                lens[i][j] = (2*np.pi/self.wavelength) * (f - pow((f**2 + ((i - self.area[0]/2)*pix_size)**2 + ((j - self.area[1]/2)*pix_size)**2), 0.5))

        for screen in range(self.N):
            self.screens.append(lens)
#Only considering one mode Gaussian->Gaussian
    def mplc(self):
        overlap_sum = 0
        for h in range(self.N): #range(N) = 0,1 (does not include N which is two in case below) 
            for m in range(len(self.input_modes)):    
                temp_in = copy.deepcopy(self.input_modes[m]) # Assignment in python does not copy objects
                temp_out = copy.deepcopy(self.output_modes[m])
                for i in range(h+1): 
                    # Step 1: Propagate to the first screen (for h = 0, range(h+1) = range(1) = 0)
                    propagate(temp_in, self.separation) 
                    if(i<h):
                        #After we have propagated we're at a screen, if this screen is before the one we're finding the overlap of, we add the phase
                        #then continue to propagate and adding phase of screens if needed
                        addPhase(temp_in, self.screens[i])
                        # Waves "meet on the left hand side of the screens, that's why we have this step"

                #Step 2: Propagating Out back through all the phase screens to the hth screen, SUBTRACTING phase where appropriate
                for j in range(self.N, h, -1):
                    propagate(temp_out, -self.separation)
                    addPhase(temp_out, -1*self.screens[j-1])
                # Now we calculate the overlap, and create the new phase from the propagation
                # Step 3: Overlap at the left hand side of the hth screen is calculated
                overlap_sum += overlap(temp_in, temp_out)
                # Step 4/5: Only one mode to consider, so we update the phase of the screen by subtracting the overlap
                # Step 6: Loops take care of going through all screens

            self.screens[h] = np.subtract(self.screens[h], np.angle(overlap_sum))
                # temp_in.field = inp.field
                # temp_out.field = outp.field
input_wave1 = wave((n,n), wavelength)
input_wave1.doubleGaussOne(1.0, 20)
# input_wave.field = initial_wave_field('/Users/ewanthomas/programing/Python/uni/Project_work/multi plane light conversion/triangle.png', input_wave)
input_wave2 = wave((n,n), wavelength)
input_wave2.doubleGaussTwo(1.0, 20)

input_modes = [input_wave1, input_wave2]

output_wave1 = wave((n,n), wavelength)
output_wave1.gaussianBeamOffCentreOne(1.0, 20)
output_wave2 = wave((n,n), wavelength)
output_wave2.gaussianBeamOffCentreTwo(1.0, 20)

output_modes = [output_wave1, output_wave2]
# output_wave.gaussianBeam(1.0, 20)
# output_wave.field = initial_wave_field('/Users/ewanthomas/programing/Python/uni/Project_work/multi plane light conversion/cat_triangle.png', output_wave)

phase_array = mulitplane_phase_array(5, int_plan_dist, input_modes, output_modes)
phase_array.initScreens(100000000000)
#phase_array.screens = np.load('/Users/ewanthomas/programing/Python/uni/Project_work/multi plane light conversion/mplc.dat.npy') #Dont forget to uncomment this and comment out the while below when you want to use data from old cycle

# i = 0
# while(i < 10):
#     phase_array.mplc()
#     i += 1

np.save('mplc.dat', phase_array.screens)

test_wave = wave((n,n), wavelength)
test_wave.doubleGaussOne(1.0, 20)
# test_wave.gaussianBeam(1.0,20)
# test_wave.field = initial_wave_field('/Users/ewanthomas/programing/Python/uni/Project_work/multi plane light conversion/triangle.png', test_wave)

test_wave2 = wave((n,n), wavelength)
test_wave2.doubleGaussOne(1.0, 20)
# test_wave2.gaussianBeam(1.0,20)
# test_wave2.field = initial_wave_field('/Users/ewanthomas/programing/Python/uni/Project_work/multi plane light conversion/triangle.png', input_wave)
dist = 1e-3 # Propagation step
animation = AngularSpectrum(test_wave, test_wave2, phase_array, 61, dist)
SaveAndDisplay(animation, "propagation_new.mp4", 30)

# for screen in range(len(phase_array.screens)):
#     plt.figure(screen)
#     plt.imshow(phase_array.screens[screen], cmap='hot')
#     plt.colorbar().set_label("Phase")
#     plt.xlabel("x (mm)")
#     plt.ylabel("y (mm)")
#     plt.title("Phase of phase screen {} in array".format(screen + 1))
#     plt.show()

