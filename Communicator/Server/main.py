import os
import sys
import latex2sympy2 as l2s2

sys.path.append(os.path.dirname("../../"))

from Latex_Extractor.main import Image2Text, convert_blackboard_image, convert_camera_image
from Calculator.main import Cal
import cv2
from Controls.main import save_bode_plot
from Integral_Transformations.fourier import Fourier
from Integral_Transformations.laplace import Laplace
from Integral_Transformations.z_transform import z_transform
from Graphing_Calculator.Graphing_calculator import plot_and_save
import matplotlib.pyplot as plt
import sympy as sp

def process_image(img_location):
    I2T = Image2Text()
    img = cv2.imread(img_location)
    # cv2.imshow("img",img)
    # cv2.waitKey(1000)
    # return(I2T.run_for_std_scenario(img))
    im2 = convert_camera_image(img)
    return(I2T.run_for_training_scenario(im2))

def process_image_for_whiteboard(img_location):
    I2T = Image2Text()
    img = cv2.imread(img_location)
    # cv2.imshow("img",img)
    # cv2.waitKey(1000)
    im2 = convert_blackboard_image(img)
    return(I2T.run_for_training_scenario(im2))

def calculate_expression(expression):
    if not expression:
        return []
    cal = Cal()
    ans = []
    for exp in expression:
        ans.append(cal.do_nothing(exp))

    return ans

def save_bode_plot_from_image(numerator,denominator):
    path = save_bode_plot(numerator,denominator)
    return path

def fourier_solver(expression,a,b):
    fourier = Fourier(expression,a,b)
    return fourier.fourier_transform

def fourier_transform_image():
    path = "fourier_spectrum.png"
    return path

def fourier_equation_image():
    path = "fourier_transform.png"
    return path

def laplace_solver(expression,a,b):
    laplace = Laplace(expression,a,b)
    return laplace.laplace_transform

def laplace_equation_image():
    path = "laplace_transform.png"
    return path

def laplace_spectrum_image():
    path = "laplace_spectrum.png"
    return path

def calculate_exp(exp:str):
    cal = Cal()
    print(exp)
    return cal.calculate(exp)

def plot_graph(exp:str):
    """
    this exp should be a latex expression
    """
    path = plot_and_save(exp)
    print(f"Plot saved at {path}")
    return path

def convert_to_sympy(expression:str):
    """
    convert the given latex string to sympy 
    expression
    """
    return l2s2.latex2sympy(expression)

def get_num_and_den(exp:str):
    exp = convert_to_sympy(exp)
    numerator = exp.as_numer_denom()[0]
    denominator = exp.as_numer_denom()[1]
    return (numerator,denominator)

def get_z_transform(expression:str):
    exp = z_transform(expression)
    #type this in latex to an image
    plt.figure(figsize=(4.3, 2), dpi=100, facecolor='black')
    plt.text(0.5, 0.5, r"$%s$" % sp.latex(exp), fontsize=24, ha='center', va='center', color='white')
    plt.axis('off')
    plt.savefig("z_transform.png")
    return "z_transform.png" 

def solve_for_x(expression:str):
    x = sp.symbols("x")
    lhs, rhs = expression.split("=")
    lhs, rhs = convert_to_sympy(lhs), convert_to_sympy(rhs) 
    eq = lhs - rhs
    ans = sp.solve(eq, x)
    if type(ans) == list:
        return ",".join(str(p) for p in ans)
    return str(ans)