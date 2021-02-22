# Title: MPLC
# Author: Ewan-James Thomas
# filename: beamby.py TODO:Rename
# License: Public Domain




#**************** Multiplane Light Conversion ****************#
#
# Purpose
# -------
#   This program uses mulitplane light conversion to map user defined input modes to user defined output modes via an 
# optimisation algorithm.
# The program allows the user to define a set of input/output electric field modes via image files (or prebuilt functions
# i.e. wave.gaussian_beam()) which are passed through the algorithm in order to find the form of the phase screen that are placed
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
import imageio
from PIL import Image


#***** Class: Wave *****#
#   This is a wave, simply an nxn array with a complex electric field value at each point that can be altered by propagation
#   
# Arguments
# ----------
# 1.) area: size of the np array
# 2.) k_0: modulus of the wavevector (=2*pi/lambda)
#
class wave():
    def __init__(self, area: np.ndarray, k_0:float):
        self.area = area
        self.field = np.zeros(area, dtype=np.complex_)
        self.k_0 = k_0
        self.wavelength = 2 * np.pi / k_0
#***** Attribute of Wave: gaussian_beam *****#
#   initialises the electric field array such that it models a gaussian beam
#
# Arguments
# ---------
# 1.) amplitude: max amplitude of the electric field
# 2.) sigma: std dev of gaussian beam aka "spread" of beam
#
    def gaussian_beam(self, amplitude: float, sigma: float):
        for i in range(self.area[0]):
            for j in range(self.area[1]):
                self.field[i][j] = amplitude * np.exp(-(((i-self.area[0]/2)*10e-6)/np.sqrt(2)*sigma)**2 -(((j-self.area[1]/2)*10e-6)/np.sqrt(2)*sigma)**2)
        #TODO: make pixel size a parameter
        #TODO: pixels separated by half a wavelength -> diffraction limit

#***** Class: Phase_array() *****#
#   
class phase_array(): #ijth value of phase array = x in exp{ix} that we multiply the field at ijth entry of wave.field
    def __init__(self, N:int, step:float, input_modes, output_modes):
        self.N = N
        self.step = step
        self.screens = []
        self.input_modes = input_modes
        self.output_modes = output_modes
        self.area = input_modes[0].area
        self.k_0 = input_modes[0].k_0
        if(len(input_modes) == len(output_modes)):
            for i in range(len(input_modes)):
                if(input_modes[i].field.size != output_modes[i].field.size):
                    raise ValueError("Input and Output must be the same size")

    def initialise_arrays(self, f:float):
        lens_profile = np.zeros(self.area, dtype=np.complex_)
        for i in range(self.area[0]):
            for j in range(self.area[1]):
                lens_profile[i][j] = self.k_0 * (f - (f**2 + ((i - self.area[0]/2)*10e-6)**2 + ((j - self.area[1]/2)*10e-6)**2)**0.5)

        for k in range(self.N):
            self.screens.append(lens_profile)


    def phase_profile(self): #arguments are a list of input/output fields i.e. input = dog and cat, output = dog_text, cat_text
        if(len(self.input_modes) == len(self.output_modes)): # Number of output modes match the input modes
            for i in range(len(self.input_modes)):
                for n in range(self.N):
                    temp_input = self.input_modes[i]
                    temp_output = self.output_modes[i]
                    for k in range(n):
                        prop(temp_input, self.step) # fields meet on the "left" side of the screen were overlaping at 
                        if(k<n): add_arbitrary_phase(temp_input, self.screens[k])
                    # Now I need to propagate the output mode backwards through the N-n screens
                    for l in range(self.N - n):
                        prop(temp_output, -self.step)
                        #the output mode is propagated back
                        add_arbitrary_phase(temp_output, self.screens[self.N-n-l-1])
                    # At this point the backwards wave is at the nth screen
                    spat_ovr = spatial_overlap(temp_input, temp_output)
                    self.screens[n] = get_phase_2(spat_ovr, self.screens[n])
            
                for f in range(self.N-1, -1, -1):
                    temp_input2 = self.input_modes[i]
                    temp_output2 = self.output_modes[i]

                    for h in range(f):
                        prop(temp_input2, self.step)
                        if(h<f): add_arbitrary_phase(temp_input2, self.screens[h])
                    for g in range(self.N - f):
                        prop(temp_output2, -self.step)
                        add_arbitrary_phase(temp_output2, self.screens[self.N - f - g -1])

                    spat_ovr2 = spatial_overlap(temp_input2, temp_output2)
                    self.screens[n] = get_phase_2(spat_ovr2, self.screens[n])

            

def get_phase_2(spatial_overlap, lens_profile):
    phase_vals = np.zeros((spatial_overlap.shape), dtype=np.complex_)
    
    for i in range(phase_vals.shape[0]):
        for j in range(phase_vals.shape[1]):
            phase_vals[i][j] = lens_profile[i][j] - np.angle(spatial_overlap[i][j])

    return phase_vals


def prop(wave, dz):
    
    kBeam = F.fft2(wave.field)
    kx = F.fftfreq(kBeam.shape[0], 1/wave.k_0)
    ky = F.fftfreq(kBeam.shape[1], 1/wave.k_0)

    for i in range(kBeam.shape[0]):
        for j in range(kBeam.shape[1]):
            kz = np.sqrt(wave.k_0*wave.k_0 -(kx[i]*kx[i]+ky[j]*ky[j]))
            kBeam[i][j] *= c.rect(1, kz*dz)

    newBeam = F.ifft2(kBeam)
    wave.field = newBeam
    return wave

def add_phase(wave, n, dz):
    for i in range(wave.area[0]):
        for j in range(wave.area[1]):
            wave.field[i][j] = wave.field[i][j] * c.rect(1, wave.k_0 * n[i][j] * dz)

    return wave

# Random elements are given refractive index n (50% of elements)
def phase_mask_random(wave, n_val: float):                  
    elements = wave.area[0] * wave.area[1]
    n = np.array([1] * (elements//2) + [n_val] * (elements//2))
    np.random.shuffle(n)
    n = np.reshape(n , (wave.area[0], wave.area[1]))

    return n
# Half of the grid has refractive index n
def phase_mask_half(wave, n_val:float):
    elements = wave.area[0] * wave.area[1]
    n = np.array([1] * (elements//2) + [n_val] * (elements//2))
    n = np.reshape(n , (wave.area[0], wave.area[1]))

    return n

def get_phase(wave):
    phase_vals = np.zeros((wave.area), dtype=np.float32)
    for i in range(wave.area[0]):
        for j in range (wave.area[1]):
            phase_vals[i][j] = np.angle(wave.field[i][j])

    return phase_vals

def add_arbitrary_phase(wave, phase_screen):
    if(wave.field.shape == phase_screen.shape):
        for i in range(wave.area[0]):
            for j in range(wave.area[1]):
                wave.field[i][j] *= c.rect(1, phase_screen[i][j])
    return wave

def spatial_overlap(input_field, output_field): #Returns the spatial overlap of the nth screen in the phase array for a certain input/output modes
        
    overlap = np.zeros((input_field.area), dtype=np.complex_)
    for i in range(input_field.area[0]):
        for j in range(input_field.area[1]):
            overlap[i][j] = input_field.field[i][j] * np.conj(output_field.field[i][j])

    return overlap

def initial_wave_field(image, wave):
    img = Image.open(image).resize((wave.area))
    img = img.convert('L')
    WIDTH, HEIGHT = img.size
    data = list(img.getdata())
    data = [data[offset:offset+WIDTH] for offset in range(0, WIDTH*HEIGHT, WIDTH)]
    data = np.asarray(data, dtype=complex)/255

    return data


newwave = wave((512,512), 2*np.pi/(500e-9)) # Initialising wave
newwave.gaussian_beam(1.0, 5000)
#newwave.field = initial_wave_field('/Users/ewanthomas/Development/Python/uni/Project_work/multi plane light conversion/triangle.png', newwave)          # Making a Gaussian lightbeam
input_list = []
input_list.append(newwave)
n = phase_mask_random(newwave, 1.2) # Initialising phase mask where half has n = 1.2

outputwave = wave((512,512),2*np.pi/(500e-9))
outputwave.gaussian_beam(1.0, 5000)
#outputwave.field = initial_wave_field('/Users/ewanthomas/Development/Python/uni/Project_work/multi plane light conversion/cat_triangle.png', outputwave)
output_list = []
output_list.append(outputwave)

pa = phase_array(4, 1500e-6, input_list, output_list)
pa.initialise_arrays(10e-3)
pa.phase_profile()

testwave = wave((512,512), 2*np.pi/(500e-9))
testwave.gaussian_beam(1,5000)
#testwave.field = initial_wave_field('/Users/ewanthomas/Development/Python/uni/Project_work/multi plane light conversion/triangle.png', newwave)

for i in range(pa.N):
    plt.figure(i)
    plt.imshow((testwave.field*np.conj(testwave.field)).real.astype("float64"), cmap='Greys')
    #plt.imshow(get_phase(newwave), cmap='hot')
    prop(testwave,(1500e-6))
    add_arbitrary_phase(testwave, pa.screens[i])
    #plt.clim(-np.pi, np.pi)
    plt.clim(0,1)
    plt.colorbar()
    plt.show()





