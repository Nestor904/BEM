#Оптимизировать работу с памятью
#Проверить код на мутации т.е. на = copy view
#проверить как ведет себя numpy в зависимости от железа, к приеру если не ватит оперативы для преобразования матриц G an F
#ОЛОЛОЛОЛОЛОЛОЛОЛОЛЛЛЛЛЛЛЛЛЛЛЛЛЛЛЛЛЛЛЛЛЛЛЛЛООООООООООООООООООО
import numpy as np
from numpy import linalg 

from scipy import integrate
from scipy import interpolate

import matplotlib.pyplot as plt

import math
from math import trunc

import timeit
from module2 import *


E=200
nu=0.28
flag_norm=True
r1=0.2
r2=1
n1=19
n2=19
stress_value=[10,0]
t0=0.5

def create_task():
    
    elements,displacement_boundary_conditions,stress_boundary_conditions,points=create_lame_task(n1=n1,n2=n2,r1=r1,r2=r2,stress_value=stress_value)
    elements=create_elements_border_interpolation(elements=elements,points=points,print_condition=0)
    elements=create_point_of_interest(elements,stress_base_function=base_function_2dlin,displacement_base_function=base_function_2dlin)
    return elements,displacement_boundary_conditions,stress_boundary_conditions,points

def create_task_from_files(path_to_model='C:\РАБОТА\Модель.txt' ,path_to_displacement_boundary_conditions='C:\РАБОТА\BCD.txt', path_to_stress_boundary_conditions='C:\РАБОТА\BCS.txt'):
    
    elements,points=read_file_model(file_path=path_to_model,print_condition=0)
    displacement_boundary_conditions={}
    displacement_boundary_conditions=read_file_boundary_conditions(file_path=path_to_displacement_boundary_conditions)
    stress_boundary_conditions={}
    stress_boundary_conditions=read_file_boundary_conditions(file_path=path_to_stress_boundary_conditions)
    elements=create_elements_border_interpolation(elements=elements,points=points,print_condition=0)
    elements=create_point_of_interest(elements,stress_base_function=base_function_2dlin,displacement_base_function=base_function_2dlin)
    return elements,displacement_boundary_conditions,stress_boundary_conditions,points

def solve_task(elements,displacement_boundary_conditions,stress_boundary_conditions,points):
 
    length_unknown_vec=count_length_unknown_vec(elements=elements,displacement_boundary_conditions=displacement_boundary_conditions,
                             stress_boundary_conditions=stress_boundary_conditions)
    evaluation_points_list=create_evaluation_points_list(length_unknown_vec=length_unknown_vec,elements=elements,points=points,t0=t0)
    F,G,displacement_key_mask,stress_key_mask=create_matrix(evaluation_points=evaluation_points_list,points=points,elements=elements,E=E,nu=nu)
    F,G,right_part,displacement_boundary_mask,stress_boundary_mask=account_border_conditions(G=G,F=F,displacement_boundary_conditions=displacement_boundary_conditions,\
        stress_boundary_conditions=stress_boundary_conditions,displacement_key_mask=displacement_key_mask,stress_key_mask=stress_key_mask,print_condition=1)


    boundary_displacement_values,boundary_stress_values=get_solution(F=F,G=G,right_part=right_part,displacement_mask=displacement_boundary_mask,stress_mask=stress_boundary_mask)

    #path="C:\\Users\\Keil\\source\\repos\\PythonApplication4\\boundary_displacement_values.txt"
    path="boundary_displacement_values.txt"
    save_results(values=list_values_to_dict(mask_num_values=displacement_key_mask,list_values=boundary_displacement_values),file_path=path)
    
    #path="C:\\Users\\Keil\\source\\repos\\PythonApplication4\\boundary_stress_values.txt"
    path="boundary_stress_values.txt"
    save_results(values=list_values_to_dict(mask_num_values=stress_key_mask,list_values=boundary_stress_values),file_path=path)

    return boundary_displacement_values,boundary_stress_values

def check_solution():
    ##проверка
    #path="C:\\Users\\Keil\\source\\repos\\PythonApplication4\\boundary_displacement_values.txt"
    #displacement_boundary_conditions=read_file_boundary_conditions(file_path=path)
    #path="C:\\Users\\Keil\\source\\repos\\PythonApplication4\\boundary_stress_values.txt"
    #stress_boundary_conditions=read_file_boundary_conditions(file_path=path)

    #F,G,displacement_key_mask,stress_key_mask=create_matrix(evaluation_points=evaluation_points_list,points=points,elements=elements,E=E,nu=nu)

    #F,G,right_part,displacement_boundary_mask,stress_boundary_mask=account_border_conditions(G=G,F=F,displacement_boundary_conditions=displacement_boundary_conditions,\
    #    stress_boundary_conditions=stress_boundary_conditions,displacement_key_mask=displacement_key_mask,stress_key_mask=stress_key_mask,print_condition=1)

    #print(right_part)
    #print("__________________________________+_+_+_+_+_+")
    return 0

#проверить нормали, с ними какая-то хрень
elements,displacement_boundary_conditions,stress_boundary_conditions,points=create_task()
#elements,displacement_boundary_conditions,stress_boundary_conditions,points=create_task_from_files()

boundary_displacement_values,boundary_stress_values=solve_task(elements=elements,displacement_boundary_conditions=displacement_boundary_conditions,\
    stress_boundary_conditions=stress_boundary_conditions,points=points)

#path="boundary_displacement_values.txt"
#boundary_displacement_values=np.array(dict_values_to_list(dict_value=read_file_boundary_conditions(file_path=path)))
#path="boundary_stress_values.txt"
#boundary_stress_values=np.array(dict_values_to_list(dict_value=read_file_boundary_conditions(file_path=path)))



print(boundary_displacement_values)
print("________________")
print(boundary_stress_values)


a=np.zeros(boundary_displacement_values.shape[0])
b=np.zeros(boundary_displacement_values.shape[0])
c=np.zeros(boundary_stress_values.shape[0])

temp=0
temp1=0
jacobian=1
gpod=collect_general_point_of_interest(elements=elements,type='border')

for i in range(0,boundary_displacement_values.shape[0]):
    point=np.array(points[gpod[i]])
    phi=np.arctan2(boundary_displacement_values[i][1],boundary_displacement_values[i][0])#-np.arctan2(point[1], point[0])
    print("______________________________")
    if abs(phi)>0.1:       
        print(i)
        print(elements[i].boundary_points[0])
        print(elements[i].boundary_points[1])
        print(points[elements[i].boundary_points[0]])
        print(points[elements[i].boundary_points[1]])
        print(np.arctan2(elements[i].evaluate_normal(points=points)[1],elements[i].evaluate_normal(points=points)[0]))
        print(boundary_displacement_values[i])
    print(str(i)+":"+str(phi))#np.arctan2(point[1], point[0])
    b[i]=phi/np.pi#linalg.norm(boundary_displacement_values[i]*np.array([math.cos(phi),math.sin(phi)]))
    a[i]=linalg.norm(boundary_displacement_values[i])
    temp=temp+a[i]

print(a)

temp1=(temp-a[0]/2-a[-1]/2)*jacobian
temp=temp/boundary_displacement_values.shape[0]

print("Среднее значение ="+str(temp))
print("Интеграл от напряжений по границе:="+str(temp1))
print("Интеграл от напряжений по границе/2pi:="+str(temp1/(2*math.pi)))

print(evaluate_displacement_at_point_in_body\
    (E=E,nu=nu,elements=elements,point_coord=[0.9,0,0],boundary_displacement_values=boundary_displacement_values,boundary_stress_values=boundary_stress_values,points=points))


for i in range(0,boundary_stress_values.shape[0]):
    c[i]=linalg.norm(boundary_stress_values[i])

print(b)

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(c[0:int(c.shape[0]/2)],label='stress at r0')
axs[0, 0].plot(c[int(c.shape[0]/2):],label='stress at r1')
plt.legend()
axs[0, 1].plot(a[0:int(a.shape[0]/2)],label='displacement at r0')
axs[0, 1].plot(a[int(a.shape[0]/2):],label='displacement at r1')
plt.legend()
axs[1, 1].plot(b[int(b.shape[0]/2):],label='angle at r1')
plt.legend()
axs[1, 0].plot(b[0:int(b.shape[0]/2)],label='angle at r0')
plt.legend()
fig.suptitle("E="+str(E)+", nu="+str(nu)+", t0="+str(t0)+ ", p1="+str(stress_value[0])+\
    ", p2="+str(stress_value[1])+", R1="+str(r1)+", R2="+str(r2)+", Flag_norm:"+str(flag_norm)+", average u: "+str(temp))
plt.show()


#for i_element in elements:
#    n=i_element.evaluate_normal(points=points) 
#    print("Элемент: "+str(i_element.number)+", угол нормали: "+str(np.angle(np.complex(-n[0],-n[1]),deg=True)))
#    print("Нормаль:"+str(n))