#ОЛОЛОЛОЛОЛОЛОЛОЛОЛЛЛЛЛЛЛЛЛЛЛЛЛЛЛЛЛЛЛЛЛЛЛЛЛООООООООООООООООООО работает ли гит?
import numpy as np
from numpy import linalg 

from scipy import integrate
from scipy import interpolate

import matplotlib.pyplot as plt

import math
from math import trunc

import timeit

def create_lame_task(n1,n2,r1,r2,stress_value=[]):
    print("Формирую сетку элементов...")
    elements=[]
    points={}
    boundary_stress_conditions={}
    boundary_displacement_conditions={}
    points1=np.zeros((n1+n2+2,3))

    def temp(r,i,n,stress_value,n_0=0):
        if i!=n+n_0:
            elements.append(element(number=i,boundary_points=[i,i+1]))
        else:
            elements.append(element(number=i,boundary_points=[i,n_0]))
        points.update({i:[r*math.cos(2*np.pi*(i-n_0)/(n+1)),r*math.sin(2*np.pi*(i-n_0)/(n+1)),0]})
        #boundary_stress_conditions.update({i:[stress_value,0]})
        boundary_stress_conditions.update({i:[stress_value*math.cos(2*np.pi*(i-n_0)/(n+1)),stress_value*math.sin(2*np.pi*(i-n_0)/(n+1))]})

    for i in range(0,n1):
        #минус идет от нормали
        temp(r=r1,i=i,n=n1,stress_value=-stress_value[0])

    if n1>0:
        temp(r=r1,i=n1,n=n1,stress_value=-stress_value[0])

    for j in range(n1+1,n1+1+n2): 
        temp(r=r2,i=j,n=n2,stress_value=stress_value[1],n_0=n1+1)

    if n2!=0:
        temp(r=r2,i=j+1,n=n2,stress_value=stress_value[1],n_0=n1+1)

    #print(boundary_stress_conditions)
    return elements,boundary_displacement_conditions,boundary_stress_conditions,points

#base_function_2dconst=[lambda x: 1]
base_function_2dlin=[lambda s: 1-s, lambda s: s]
#t3=0
#t4=1-t3
#base_function_2dlin=[lambda s: math.cos(math.pi/2*(s-t3)/(t4-t3)), lambda s: math.sin(math.pi/2*(s-t3)/(t4-t3))]
#base_function_2dlin=[lambda s: 0.5*(1+s), lambda s:  0.5*(1-s)]

#В случае решения мнгозоновых задач, создать класс body{


#в коорый вошли бы список гр. элементов, характеристики тела, и некотрые функции относящиеся к формированию тела,
# такие как задача граничных условий, считывание и т.д
class element:
    def __init__(self,number=None,boundary_points=[], points_of_displacement=[], points_of_stress=[], additional_geometry_points=[]):
        self.number=number
        self.dimm=2
        self.boundary_points = boundary_points
        self.points_of_displacement = points_of_displacement
        self.points_of_stress = points_of_stress
        self.additional_geometry_points = additional_geometry_points
        self.interpolant=None
        self.evaluation_points={}
        
    
    def create_border_interpolation_function(self,points):
        self.temp_num_points=self.boundary_points 
        self.temp_num_points[1:1]=self.additional_geometry_points
        self.coords=np.array([points[num] for num in self.temp_num_points])
        if len(self.temp_num_points)<=5:
            self.interpolant=interpolate.splprep([self.coords[:,0],self.coords[:,1]],k=len(self.temp_num_points)-1)[0]
        else:
            self.interpolant=interpolate.splprep([self.coords[:,0],self.coords[:,1]],k=3)[0]
        #k!!!!! modify

    #t=0 borderpont[0], t=1 borderpoint[1]
    def evaluate_border_point(self,t=0,derivative=0):
        if self.interpolant==None:
            print("Create border interpolation function for element №" + str(self.number))
            return [0,0]
        z=0

        return interpolate.splev(t,self.interpolant,derivative)+[np.array(z)]
        
 
    def create_base_function(self,base_function,type ):
    #проверить размерность функций и кол-ва точек
        if type=='stress':
            self.stress_base_functions={self.points_of_stress[i]:base_function[i] for i in range(0,len(self.points_of_stress))}
            return
        if type=='displacement':
            self.displacement_base_functions={self.points_of_displacement[i]:base_function[i] for i in range(0,len(self.points_of_displacement))}
            return
        else:
            print("Error type for element №" +str(self.number)+ ". Type can be stress or displacement")
#}

    def evaluate_jacobian(self,points,t=0.5):
        
        #jacobian=linalg.norm(np.array(points[self.boundary_points[1]])-np.array(points[self.boundary_points[0]]))
        return linalg.norm(self.evaluate_border_point(t,derivative=1))
        #return jacobian

    def evaluate_normal(self,points,t=0.5, flag_norm=True):
        kx,ky,kz=(np.array(points[self.boundary_points[1]])-np.array(points[self.boundary_points[0]]))
    
        #if flag_norm:
        norm=linalg.norm(np.array(points[self.boundary_points[1]])-np.array(points[self.boundary_points[0]]))
        #else:
        #    norm=1

        return -ky/norm, kx/norm, kz/norm

def evaluate_singular_integral_of_fundumental_stress_solution(E,nu,i_element,evaluate_point,points):
    G=E/(2*(1+nu)) 
    C1=-1/(8*np.pi*G*(1-nu))
    C2=3-4*nu

    evaluate_point_coords=points[evaluate_point]
    x0,y0,z0=evaluate_point_coords

    kx,ky,kz=np.array(points[i_element.boundary_points[1]])-np.array(points[i_element.boundary_points[0]])
    bx,by,bz=np.array(points[i_element.boundary_points[0]])
    
    t0=i_element.evaluation_points[evaluate_point]
    jacobian=i_element.evaluate_jacobian(points=points)
 
    #Simplified non-singular part of fundamental solution for linear element, x=k_x*t+b_x, y=k_y*t+b_y  
    temp=-C1*kx*ky/(kx*kx+ky*ky)
    temp1=C1*(C2*(0.5)*math.log(kx*kx+ky*ky)-kx*kx/(ky*ky+kx*kx))
    temp2=C1*(C2*(0.5)*math.log(kx*kx+ky*ky)-ky*ky/(ky*ky+kx*kx))
    integrated_stress_solution={}
    #проверить alg-logb, могут быть не точности со знаками t->-t, проверка показала, что все норм
    for num_point in i_element.points_of_stress:
        temp3=+integrate.quad(lambda t: jacobian*C1*C2*i_element.stress_base_functions[num_point](t+t0) ,-t0,0,weight='alg-logb',wvar=[0,0])[0]\
              +integrate.quad(lambda t: jacobian*C1*C2*i_element.stress_base_functions[num_point](t),t0,1,weight='alg-loga',wvar=[0,0])[0]
        integrated_stress_solution.update({num_point:np.zeros([i_element.dimm,i_element.dimm])})
        integrated_stress_solution[num_point][0,1]=integrate.quad(lambda t: jacobian*temp*i_element.stress_base_functions[num_point](t),0,1)[0]
        integrated_stress_solution[num_point][1,0]=integrate.quad(lambda t: jacobian*temp*i_element.stress_base_functions[num_point](t),0,1)[0]
        integrated_stress_solution[num_point][0,0]=integrate.quad(lambda t: jacobian*temp1*i_element.stress_base_functions[num_point](t),0,1)[0]+temp3
        integrated_stress_solution[num_point][1,1]=integrate.quad(lambda t: jacobian*temp2*i_element.stress_base_functions[num_point](t),0,1)[0]+temp3

    return integrated_stress_solution

def evaluate_regular_integral_of_fundumental_stress_solution(E,nu,i_element,evaluate_point_coords, points):
        G=E/(2*(1+nu)) 
        C1=-1/(8*np.pi*G*(1-nu))
        C2=3-4*nu
        integrated_stress_solution ={}
        jacobian=i_element.evaluate_jacobian(points=points)
        def fundumental_stress_solution(t,i=0,j=0):
            coords=np.array(i_element.evaluate_border_point(t=t,derivative=0))-np.array(evaluate_point_coords)
            r=linalg.norm(coords)
            return (C1*(C2*np.eye(2)[i,j]*math.log(r)-coords[i]*coords[j]/(r*r)))
         
        for num_point in i_element.points_of_stress:    
            integrated_stress_solution.update({num_point:np.zeros([i_element.dimm,i_element.dimm])})
            integrated_stress_solution[num_point][0,1]=integrate.quad(lambda t: jacobian*fundumental_stress_solution(t,i=0,j=1)*i_element.stress_base_functions[num_point](t),0,1)[0]
            integrated_stress_solution[num_point][1,0]=integrate.quad(lambda t: jacobian*fundumental_stress_solution(t,i=1,j=0)*i_element.stress_base_functions[num_point](t),0,1)[0]
            integrated_stress_solution[num_point][0,0]=integrate.quad(lambda t: jacobian*fundumental_stress_solution(t,i=0,j=0)*i_element.stress_base_functions[num_point](t),0,1)[0]
            integrated_stress_solution[num_point][1,1]=integrate.quad(lambda t: jacobian*fundumental_stress_solution(t,i=1,j=1)*i_element.stress_base_functions[num_point](t),0,1)[0]

        return  integrated_stress_solution

def integrate_fundumental_stress_solution(E,nu, i_element, evaluate_point,points):
    #E модуль сдвига 
    #nu модуль Пуассона

    if evaluate_point in i_element.evaluation_points:
        #print(str(evaluate_point)+" №"+str(i_element.number))
        #технчески в качестве входного параметра можно указать две функции отвечающие за интегрирование синг части и нормальной
        integrated_stress_solution= evaluate_singular_integral_of_fundumental_stress_solution(E=E,nu=nu,\
            i_element=i_element,evaluate_point=evaluate_point,points=points)
    else:
        integrated_stress_solution= evaluate_regular_integral_of_fundumental_stress_solution(E=E,nu=nu,\
            i_element=i_element,evaluate_point_coords=points[evaluate_point],points=points)


    return integrated_stress_solution

def evaluate_singular_integral_of_fundumental_displacement_solution(E,nu, i_element, evaluate_point,points):
    #E модуль сдвига 
    #nu модуль Пуассона
    C3=-1/(4*np.pi*(1-nu))
    C4=1-2*nu

    evaluate_point_coords=points[evaluate_point]

    kx,ky,kz=(np.array(points[i_element.boundary_points[1]])-np.array(points[i_element.boundary_points[0]]))
    bx,by,bz=np.array(points[i_element.boundary_points[0]])
   
    nx,ny,nz=i_element.evaluate_normal(points=points)
    
    #Simplified non-singular part of fundamental solution for linear element, x=k_x*t+b_x, y=k_y*t+b_y
    t0=i_element.evaluation_points[evaluate_point]
    jacobian=i_element.evaluate_jacobian(points=points)

    temp00=C3*C4*(kx*nx+ky*ny)/(kx*kx+ky*ky)+2*C3*(kx*kx*kx*nx+kx*kx*ky*ny)/((kx*kx+ky*ky)*(kx*kx+ky*ky))
    temp01=C3*C4*(kx*ny-ky*nx)/(kx*kx+ky*ky)+2*C3*(kx*ky*kx*nx+ky*kx*ky*ny)/((kx*kx+ky*ky)*(kx*kx+ky*ky))
    temp10=C3*C4*(ky*nx-kx*ny)/(kx*kx+ky*ky)+2*C3*(ky*kx*kx*nx+ky*kx*ky*ny)/((kx*kx+ky*ky)*(kx*kx+ky*ky))
    #temp10=C3*C4*(kx*ny-ky*nx)/(kx*kx+ky*ky)+2*C3*(kx*kx*ky*nx+ky*kx*ky*ny)/((kx*kx+ky*ky)*(kx*kx+ky*ky))
    #temp01=C3*C4*(ky*nx-kx*ny)/(kx*kx+ky*ky)+2*C3*(kx*kx*ky*nx+ky*kx*ky*ny)/((kx*kx+ky*ky)*(kx*kx+ky*ky))
    temp11=C3*C4*(kx*nx+ky*ny)/(kx*kx+ky*ky)+2*C3*(ky*ky*kx*nx+ky*ky*ky*ny)/((kx*kx+ky*ky)*(kx*kx+ky*ky))

    integrated_displacement_solution={}
    for num_point in i_element.points_of_displacement:    
        integrated_displacement_solution.update({num_point:np.zeros([i_element.dimm,i_element.dimm])})
        integrated_displacement_solution[num_point][0,1]=integrate.quad(lambda t:jacobian*temp01*i_element.displacement_base_functions[num_point](t),0,1,weight='cauchy',wvar=t0)[0]
        integrated_displacement_solution[num_point][1,0]=integrate.quad(lambda t:jacobian*temp10*i_element.displacement_base_functions[num_point](t),0,1,weight='cauchy',wvar=t0)[0]
        integrated_displacement_solution[num_point][0,0]=integrate.quad(lambda t:jacobian*temp00*i_element.displacement_base_functions[num_point](t),0,1,weight='cauchy',wvar=t0)[0]
        integrated_displacement_solution[num_point][1,1]=integrate.quad(lambda t:jacobian*temp11*i_element.displacement_base_functions[num_point](t),0,1,weight='cauchy',wvar=t0)[0]
            
    return integrated_displacement_solution

def evaluate_regular_integral_of_fundumental_displacement_solution(E,nu,i_element,evaluate_point_coords,points):
    #E модуль сдвига 
    #nu модуль Пуассона
    C3=-1/(4*np.pi*(1-nu))
    C4=1-2*nu

    
    kx,ky,kz=np.array(points[i_element.boundary_points[1]])-np.array(points[i_element.boundary_points[0]])
    n=i_element.evaluate_normal(points=points)
    jacobian=i_element.evaluate_jacobian(points=points)
    
    integrated_displacement_solution={}

    def fundumental_displacement_solution(t,i=0,k=0):
        coords=np.array(i_element.evaluate_border_point(t=t,derivative=0))-np.array(evaluate_point_coords)
        r=linalg.norm(coords)

        return (C3/(r*r))*(C4*(n[k]*coords[i]-n[i]*coords[k])+(C4*np.eye(2)[i,k]+2*coords[i]*coords[k]/(r*r))*(coords[0]*n[0]+coords[1]*n[1]))

    for num_point in i_element.points_of_displacement:    
        integrated_displacement_solution.update({num_point:np.zeros([i_element.dimm,i_element.dimm])})
        integrated_displacement_solution[num_point][0,1]=(integrate.quad(lambda t: jacobian*fundumental_displacement_solution(t=t,i=0,k=1)*i_element.displacement_base_functions[num_point](t),0,1)[0])
        integrated_displacement_solution[num_point][1,0]=(integrate.quad(lambda t: jacobian*fundumental_displacement_solution(t=t,i=1,k=0)*i_element.displacement_base_functions[num_point](t),0,1)[0])
        integrated_displacement_solution[num_point][0,0]=(integrate.quad(lambda t: jacobian*fundumental_displacement_solution(t=t,i=0,k=0)*i_element.displacement_base_functions[num_point](t),0,1)[0])
        integrated_displacement_solution[num_point][1,1]=(integrate.quad(lambda t: jacobian*fundumental_displacement_solution(t=t,i=1,k=1)*i_element.displacement_base_functions[num_point](t),0,1)[0])

    return  integrated_displacement_solution

def account_angle(E,nu,i_element,points,evaluate_point):
    C3=-1/(4*np.pi*(1-nu))
    C4=1-2*nu
    jacobian=i_element.evaluate_jacobian(points=points)
    t0=i_element.evaluation_points[evaluate_point]
    
    angle=np.pi
    a=np.eye(2)*(-1-C3*((C4+1)*angle+math.sin(2*angle)/2))+(np.eye(2)-1)*math.sin(angle)**2
    a[1,1]=a[1,1]+C3*math.sin(2*angle)

    temp={}
    for pod in i_element.points_of_displacement:
        temp.update({pod:np.zeros([i_element.dimm,i_element.dimm])})
        temp[pod][0,0]=jacobian*a[0,0]*i_element.displacement_base_functions[pod](t0)
        temp[pod][1,1]=jacobian*a[1,1]*i_element.displacement_base_functions[pod](t0)
        temp[pod][0,1]=jacobian*a[0,1]*i_element.displacement_base_functions[pod](t0)
        temp[pod][1,0]=jacobian*a[1,0]*i_element.displacement_base_functions[pod](t0)
    return temp

def integrate_fundumental_displacement_solution(E,nu, i_element, evaluate_point,points):
    #E модуль сдвига 
    #nu модуль Пуассона
    #технчески в качестве входного параметра можно указать две функции отвечающие за интегрирование синг части и нормальной
    if evaluate_point in i_element.evaluation_points:
        #print(str(evaluate_point)+" №"+str(i_element.number))
        integrated_displacement_solution= evaluate_singular_integral_of_fundumental_displacement_solution(E=E,nu=nu,\
            i_element=i_element,evaluate_point=evaluate_point,points=points)
        accounted_angle_part=account_angle(E=E,nu=nu,i_element=i_element,points=points,evaluate_point=evaluate_point)
        for pod in i_element.points_of_displacement:
            integrated_displacement_solution[pod]=integrated_displacement_solution[pod]+accounted_angle_part[pod]
    else:
        integrated_displacement_solution= evaluate_regular_integral_of_fundumental_displacement_solution(E=E,nu=nu,\
            i_element=i_element,evaluate_point_coords=points[evaluate_point],points=points)


    return integrated_displacement_solution

#}
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False

def read_file_model(file_path, print_condition=0):
#считывание обыкновенного файла 
#обработка номеров элементов, первый список в элементе это список граничных точек элемента, второй список это список узлов в которых будут вычисляться напряжения и перемещения 
    print("Пытаюсь открыть файл с моделью....")
    file_stream = open(file_path) 
    print("Считываю данные....")
    file_info=file_stream.read().split('\n')
    file_stream.close()
    num_elements=int(file_info[1]) 
    elements=file_info[2:num_elements+2:1]
    print("Формирую граничные точки элементов....") 
    for i in range(0,num_elements): 
        elements[i]=element(number=i,boundary_points=list(map(int,elements[i].split('|')[0].split(' '))))

    #обработка точек, каждому номеру точки соответствует координаты в формате XYZ 
    print("Делаю словарь координат точек....")
    num_points=int(file_info[2+1+num_elements]) 
    points={i:file_info[2+2+num_elements+i].split(' ') for i in range(0, num_points)} 
    for point in points: 
        for coordinate in range(0,3): 
            points[point][coordinate]=float(points[point][coordinate]) 
    print("Успешно! Возвращаю список элементов и точек")
    if print_condition!=0:
        print(points)
        for i_element in elements:
            print(i_element.boundary_points)
    return [elements,points]

def read_file_boundary_conditions(file_path,print_condition=1):
    #считывание граничных условий
    #номер точки: значение #если "x" none. то значение неизвестно
    print("Открываю файл по пути " + str(file_path))
    file_stream = open(file_path) 
    print("Считываю файл...")
    file_info=file_stream.read().split('\n') 
    file_stream.close()
    print("Формирую словарь граничных условий...")
    num_conditions=int(file_info.pop(0))
    conditions={int(file_info[i].split(':')[0]):list(map(lambda x: float(x) if is_number(x) else x,file_info[i].split(':')[1].split(' '))) for i in range(0, num_conditions)} 
    print("Успешно! Возврашаю граничные условия")
    if print_condition!=0:
        print(conditions)
    return conditions

def collect_general_point_of_interest(elements,type):
    if type=='stress':  
        general_points_of_stress=[]
        for ielement in elements:     
            general_points_of_stress.extend(ielement.points_of_stress)
        general_points_of_stress=list(set(general_points_of_stress))
        return general_points_of_stress
    if type=='border':  
        general_points_of_border=[]
        for ielement in elements:     
            general_points_of_border.extend(ielement.boundary_points)
        general_points_of_border=list(set(general_points_of_border))
        return general_points_of_border
    if type=='displacement':  
        general_points_of_displacement=[]
        for ielement in elements:     
            general_points_of_displacement.extend(ielement.points_of_displacement)
        general_points_of_displacement=list(set(general_points_of_displacement))
        return general_points_of_displacement
    else:
        print("Error type in collect_general_point_of_interest")

def count_length_unknown_vec(elements,displacement_boundary_conditions,stress_boundary_conditions):
    #вычисление кол-ва точек, в которых будут записываться интегральные уравнения
    length_unknown_vec=0
    
    general_points_of_displacement=collect_general_point_of_interest(elements=elements,type='displacement')
    general_points_of_stress=collect_general_point_of_interest(elements=elements,type='stress')

    length_unknown_vec=0
    for i_element in elements:
        for point in i_element.points_of_displacement:
            if point in general_points_of_displacement:
                #if point in double_point: dont remove from gpod
                general_points_of_displacement.remove(point)
                length_unknown_vec=length_unknown_vec+i_element.dimm
        for point in i_element.points_of_stress:
            if point in general_points_of_stress:
                general_points_of_stress.remove(point)
                length_unknown_vec=length_unknown_vec+i_element.dimm
   
    ##количество известных перемещений
    for key in displacement_boundary_conditions.keys():
        length_unknown_vec= length_unknown_vec-len(displacement_boundary_conditions[key])+displacement_boundary_conditions[key].count("x")
    ##количество известных напряжений
    for key in stress_boundary_conditions.keys():
        length_unknown_vec= length_unknown_vec-len(stress_boundary_conditions[key])+stress_boundary_conditions[key].count("x")
    
    return length_unknown_vec

def create_elements_border_interpolation(elements,points,print_condition):
    print("Строю сплайны для граничных элементов...")
    for i_element in elements:       
        i_element.create_border_interpolation_function(points=points)
        if print_condition!=0:
            print("Элемент №"+str(i_element.number)+" Успешно!")
    print("Успешно! Сплайны построены, возвращаю обновленный список элементов")
    return elements

def create_point_of_interest(elements,stress_base_function,displacement_base_function,print_condition=0):
    #различные виды формирования точек интереса ил скорее задание типа элемента
    print("Формирую точки, в которых вычисляются перемещения и деформации, и базовые функции на элементе...")
    for i_element in elements:
        i_element.points_of_displacement=i_element.boundary_points    
        i_element.create_base_function(displacement_base_function,type='displacement')
        i_element.points_of_stress=i_element.boundary_points
        i_element.create_base_function(stress_base_function,type='stress')
        if print_condition!=0:
            print("Точки и базовые функции успешно построены для элемента №"+str(i_element.number))
    print("Успешно! Возвращаю обновленный список элементов")
    return elements
 
#стоит учесть работу с 3d, если одна из координат постоянна
def create_evaluation_points(id_point,i_element, num_evaluation_points ,points,alpha,t0):
    z=0#!!!!!!!
    evaluation_points=[]
    if num_evaluation_points==1:
        points.update({id_point:i_element.evaluate_border_point(t=t0)})
        i_element.evaluation_points.update({id_point:t0})
        evaluation_points.extend([id_point])
    else:
          h=(1-2*alpha)/(num_evaluation_points-1)
          new_points={id_point+i:i_element.evaluate_border_point(t=alpha+h*i) for i in range(0,num_evaluation_points)}
          points.update(new_points)
          i_element.evaluation_points.update({id_point+i:alpha+h*i for i in range(0,num_evaluation_points)})
          evaluation_points.extend(list(new_points.keys()))

    return evaluation_points

def create_evaluation_points_list( length_unknown_vec,elements, points, alpha=0.25,t0=0.001, print_condition=1):   
    #обдумать другой способ формирования, в котором точки будут расплагаться более равномерно
    print("Формирую список точек, в которых будут записываться интегральные соотношения...")
    print("Общее количество неизвестных --- " + str(length_unknown_vec))
    evaluation_points=[]
    #список элементов в которых нет граничных условий
    #temp=[element.number for element in elements
    #    if len(list(set(element.points_of_displacement)&set(list(u.keys())))) + len(list(set(element.points_of_stress)&set(list(t.keys())))) ==0 ]
 
    num_evaluation_points=0
    i=0
    dimm=2
    
    a=divmod(length_unknown_vec,dimm)[0]
    print("Количество векторных неизвестных --- " +str(a)+" с учетом размерности пространства: " + str(dimm))
    b=divmod(a,len(elements))[0]
    c=divmod(a,len(elements))[1]
    print("Количество элементов: "+str(len(elements)-c)+" , в которых будет размещено: "+str(b)+" точек, в которых записывается интегральное соотношение")
    print("Количество элементов: "+str(c)+" , в которых будет размещено: "+str(b+1)+" точек, в которых записывается интегральное соотношение")
    
    id_point=max(points.keys())+1
    print("Начинаю размещать точки...")
    for i_element in elements:
        if (c==0)and(b==0):
            break
        else:
            if c!=0:         
                evaluation_points.extend(create_evaluation_points(id_point=id_point,alpha=alpha,t0=t0,i_element=i_element,num_evaluation_points=b+1,points=points))
                id_point=id_point+b+1
                c=c-1 
            else:  
                evaluation_points.extend(create_evaluation_points(id_point=id_point,alpha=alpha,t0=t0,i_element=i_element,num_evaluation_points=b,points=points))
                id_point=id_point+b
    print("Успешно! Возвращаю список точек для вычисления интегральных соотношений")
    if print_condition!=0:
        print(evaluation_points)
    return evaluation_points

def create_matrix(evaluation_points,points,elements,E,nu,print_condition=0):
    #проверить знаки в интегрировании
    #F - проинтерированные фундаментальные соотношения стоящие при перемещениях
    #G - проинтерированные фундаментальные соотношения стоящие при напряжениях
    print("Формирую матрицы F и G...")
    dimm=2
    general_points_of_displacement=collect_general_point_of_interest(elements=elements,type='displacement')
    general_points_of_stress=collect_general_point_of_interest(elements=elements,type='stress')
    F=np.zeros((len(evaluation_points),len(general_points_of_displacement),2,2))
    G=np.zeros((len(evaluation_points),len(general_points_of_stress),2,2))
    print("Размерность матрицы F: " + str(F.shape))
    print("Размерность матрицы G: " + str(G.shape))

    g=timeit.default_timer()
    tmp=min(evaluation_points)

    displacement_key_mask={}
    stress_key_mask={}
    i=0
    for pod in general_points_of_displacement:
        displacement_key_mask.update({pod:i})
        i=i+1
    i=0
    for pos in general_points_of_stress:
        stress_key_mask.update({pos:i})
        i=i+1


    print("Начинаю заполнять матрицы F и G...")
    for e_point in evaluation_points:          
        print("Номер точки вычисления: "+str(e_point))
        for i_element in elements:
            #print("Элемент № "+str(i_element.number))
            #print("Вычисляю локальную матрицу F")
            integrated_fundamental_displacement_solution=integrate_fundumental_displacement_solution(E=E,nu=nu,\
                i_element=i_element,evaluate_point=e_point,points=points)
            #print(integrated_fundamental_displacement_solution)
            for point_of_displacement in integrated_fundamental_displacement_solution.keys():             
                F[e_point-tmp,displacement_key_mask[point_of_displacement]]=F[e_point-tmp,displacement_key_mask[point_of_displacement]]+integrated_fundamental_displacement_solution[point_of_displacement]
            #print("Вычисляю локальную матрицу G")         
            integrated_fundamental_stress_solution=integrate_fundumental_stress_solution(E=E,nu=nu,\
                i_element=i_element,evaluate_point=e_point,points=points)
            #print(integrated_fundamental_displacement_solution)
            for point_of_stress in integrated_fundamental_stress_solution.keys():             
                G[e_point-tmp,stress_key_mask[point_of_stress]]=G[e_point-tmp,stress_key_mask[point_of_stress]]+integrated_fundamental_stress_solution[point_of_stress]
    print("Готово! Общее время выполнения: " + str(timeit.default_timer()-g))

    #в идеале заменить
    print("Преобразую матрицы из блочного вида")
    n1,n2,n3,n4=G.shape
    G=G.reshape(n1,n2,-1,n3,n4).swapaxes(1,3).reshape(n1*n3,n2*n4)
    n1,n2,n3,n4=F.shape
    F=F.reshape(n1,n2,-1,n3,n4).swapaxes(1,3).reshape(n1*n3,n2*n4)

    #n1,n2,n3,n4=arr.shape
    #arr.reshape(n1,n2,-1,n3,n4).swapaxes(1,3).reshape(n1*n3,n2*n4)
    print("Готово! Возвращаю матрицы F и G")
    return F,G,displacement_key_mask,stress_key_mask

def get_solution(F,G,right_part,displacement_mask ,stress_mask):
    print("Обьединяю матрицы F и G...")
    matrix=np.concatenate((F,-G),axis=1)
    print("Размерность обьединенной матрицы:" + str(matrix.shape))
    print("Решаю СЛАУ...")
    solution=linalg.solve(matrix,-right_part)
    #проверить шейп и следование неизвестных, или же их ориентацию
    print("Подставляю в полученное решение для перемещений известные значения...")
    #сортировка обязательна
    temp=list(displacement_mask.keys())
    temp.sort()
    for key in temp:
        solution=np.insert(solution,key,displacement_mask[key])
    print("Чистое решение:")
    print(solution)
    boundary_displacement_values=solution[0:F.shape[1]+len(list(displacement_mask.keys()))]
    boundary_displacement_values=boundary_displacement_values.reshape(int(boundary_displacement_values.shape[0]/2),2)

    boundary_stress_values=solution[F.shape[1]+len(list(displacement_mask.keys())):]
    print("Подставляю в полученное решение для напряженний известные значения...")
    temp=list(stress_mask.keys())
    temp.sort()
    for key in temp:
        boundary_stress_values=np.insert(boundary_stress_values,key,stress_mask[key])
    boundary_stress_values=boundary_stress_values.reshape(int(boundary_stress_values.shape[0]/2),2)

    print("Готово! Возвращаю перемещения и напряжения на границе")
    return[boundary_displacement_values,boundary_stress_values]

def account_border_conditions(F,G,displacement_boundary_conditions,stress_boundary_conditions,displacement_key_mask,stress_key_mask,print_condition=0):
    #размерности!!!!!
    print("Учитываю граничные условия...")
    known_values=np.zeros(F.shape[0])
    dimm=2


    print("Учитываю граничные условия для перемещений...")
    displacement_boundary_mask={}

    temp=list(displacement_boundary_conditions.keys())

    if print_condition!=0:
        print("Номера точек в которых заданы перемещения...")
        print(temp)
    
    for key in temp:
        if print_condition!=0:
            print("Номер точки: " + str(key) +" , граничное условие: " + str(displacement_boundary_conditions[key]))
        i=0
        for condition in displacement_boundary_conditions[key]:
            if print_condition!=0:
                print("Позиция, учитываемого граничного условия в матрице: " + str(displacement_key_mask[key]+i))
            if is_number(condition):
                known_values=known_values+condition*F[:,displacement_key_mask[key]*dimm+i]
                displacement_boundary_mask.update({displacement_key_mask[key]*dimm+i:condition})                                         
            i=i+1
    F=np.delete(F,list(displacement_boundary_mask.keys()),1)

 
    print("Учитываю граничные условия для напряжений...")
    stress_boundary_mask={}

    temp=list(stress_boundary_conditions.keys())

    if print_condition!=0:
        print("Отсортированные по возрастанию номера точек, в которых заданы напряжения...")
        print(temp)
    for key in temp:
        if print_condition!=0:
            print("Номер точки: " + str(key) +" , граничное условие: " + str(stress_boundary_conditions[key]))
        i=0
        for condition in stress_boundary_conditions[key]:
            if print_condition!=0:
                print("Позиция, учитываемого граничного условия в матрице: " + str(stress_key_mask[key]*dimm+i))
            if is_number(condition):
                known_values=known_values-condition*G[:,stress_key_mask[key]*dimm+i]
                stress_boundary_mask.update({stress_key_mask[key]*dimm+i:condition})  
            i=i+1
    G=np.delete(G,list(stress_boundary_mask.keys()),1)
    print("Размерность матрицы F:"+str(F.shape))
    print("Размерность матрицы G:"+str(G.shape))
    print("Успешно! Возвращаю значения матриц F,G, правой части и номера удаленных столбцов перемещений и напряженний")
    return F,G,known_values,displacement_boundary_mask,stress_boundary_mask
  
def evaluate_displacement_at_point_in_body(E,nu,elements,point_coord,boundary_displacement_values,boundary_stress_values,points):
    dimm=2
    displacement=np.zeros(dimm)
    for i_element in elements:
        integrated_solution=evaluate_regular_integral_of_fundumental_displacement_solution(E=E,nu=nu,i_element=i_element,evaluate_point_coords=point_coord,points=points)
        for point in i_element.points_of_displacement:
            displacement=displacement+np.inner(integrated_solution[point],boundary_displacement_values[point])
## осторожно с минусами
        integrated_solution=evaluate_regular_integral_of_fundumental_stress_solution(E=E,nu=nu,i_element=i_element,evaluate_point_coords=point_coord,points=points)
        for point in i_element.points_of_stress:
            displacement=displacement-np.inner(integrated_solution[point],boundary_stress_values[point])
    return displacement

def check_geometry(elements,points):
    temp=np.zeros((len(elements),3))
    i=0
    for i_element in elements:
        t=0
        temp[i][0]=linalg.norm(i_element.evaluate_border_point(t=t)-np.array(points[i_element.boundary_points[t]]))
        t=1
        temp[i][1]=linalg.norm(i_element.evaluate_border_point(t=t)-np.array(points[i_element.boundary_points[t]]))
        t=0.5
        temp[i][1]=linalg.norm(i_element.evaluate_border_point(t=t)-(np.array(points[i_element.boundary_points[1]])-np.array(points[i_element.boundary_points[0]]))*t+np.array(points[i_element.boundary_points[0]]))
        i=i+1
        print(temp)
        plt.plot(temp.transpose()[0,:], label="разница в координатах для t=0")
        plt.plot(temp.transpose()[2,:], label="разница в координатах для t=0.5")
        plt.plot(temp.transpose()[1,:], label="разница в координатах для t=1")
        plt.legend()
        plt.show()
    return

def save_results(file_path,type='w',values={}):
    #type a-дозапись,w - перезаписывает
    file_stream = open(file_path,type)
    file_stream.write(str(len(values.keys()))+"\n")
    for key in values.keys():
        #print(str(key)+":"+str(values[key][0])+" "+str(values[key][1])+"\n")
        file_stream.write(str(key)+":"+str(values[key][0])+" "+str(values[key][1])+"\n")
    file_stream.close()

def list_values_to_dict(list_values,mask_num_values={}):
    dict_value={}
    if len(list(mask_num_values))==0:
        mask_num_values={i:i for i in range(0,len(list_values))}
    for key in mask_num_values:
        dict_value.update({key:list_values[mask_num_values[key]]})
    return dict_value

def dict_values_to_list(dict_value,mask_num_values={}):
    list_value=[0 for i in range(0,len(dict_value.keys()))]
    if len(list(mask_num_values.keys()))==0:
        mask_num_values={i:i for i in range(0,len(list_value))}
    for key in mask_num_values:
        list_value[mask_num_values[key]]=dict_value[key]
    return list_value
