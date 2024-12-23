#Importacion bibliotecas

import numpy as np
import copy
import re
import math
import time
import os
import matplotlib.pyplot as plt
import numpy as np

from random import shuffle, random, randint

#Declaracion de funciones

def orders_splitting(data, v_t):
    p = r'[\d]+[.,\d]+|[\d]*[.][\d]+|[\d]+'
    l = r'[a-zA-Z]+'
    i = 0
    for text in data:
        if text == '-':
            i += 1
            continue
        data[i] = text.split("+")
        i += 1
    i = 0
    for orders in data:
        if orders == '-':
            i += 1
            continue
        j = 0
        for order in orders:
            quantity = re.findall(p, order)
            if len(quantity) == 0:
                quantity = ['1']
            type = re.findall(l, order)
            data[i][j] = [int(quantity[0]), ord(type[0]) - 65]
            j += 1
        i += 1
    cont = 0
    for i in data:
        if i == '-':
            cont += 1
            continue
        t_l = 0
        t_w = 0
        for j in i:
            t_l += j[0] * v_t[1][j[1]] / 1000
            t_w += j[0] * v_t[4][j[1]] / 1000
        data[cont] = [t_l, t_w]
        cont += 1
    return data

def lectura_archivos(file_transports, file_finished, file_customers, file_parameters, file_pop, carpeta_data, numero_clientes):
    #Importacion y tratamientos de txt
    file_customers = f"customer_order_numero_{file_customers[0]}_limite_{file_customers[1]}.txt"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dir_comp = os.path.join(current_dir, 'data_compartida')
    dir_data = os.path.join(current_dir, carpeta_data)
    dir_ordenes_generadas = os.path.join(current_dir, 'ordenes_generadas')
    archivo_1 = os.path.join(dir_data, file_transports)
    archivo_2 = os.path.join(dir_comp, file_finished)
    archivo_3 = os.path.join(dir_ordenes_generadas, file_customers)
    archivo_4 = os.path.join(dir_comp,file_parameters)
    archivo_5 = os.path.join(dir_comp,file_pop)

    trasport_vehicle_data = []
    finished_vehicle_data = []
    customer_orders_information_with_time_windows = []
    parameters = []
    poblation_data = []  
    
    file = open(archivo_1, 'r')
    contents = file.read()
    file.close
    transport_VT = []
    vehicle_typenum = []
    loading_lenght = []
    loading_widht = []
    loading_height = []
    loading_weight = []
    fixed_cost = []
    selfweight = []
    for line in contents.split('\n'):
        row = line.split(' ')
        transport_VT.append(int(row[0]))
        vehicle_typenum.append(int(row[1]))
        loading_lenght.append(float(row[2]))
        loading_widht.append(float(row[3]))
        loading_height.append(float(row[4]))
        loading_weight.append(float(row[5]))
        fixed_cost.append(int(row[6]))
        selfweight.append(float(row[7]))
    trasport_vehicle_data.append(transport_VT)
    trasport_vehicle_data.append(vehicle_typenum)
    trasport_vehicle_data.append(loading_lenght)
    trasport_vehicle_data.append(loading_widht)
    trasport_vehicle_data.append(loading_height)
    trasport_vehicle_data.append(loading_weight)
    trasport_vehicle_data.append(fixed_cost)
    trasport_vehicle_data.append(selfweight)

    file = open(archivo_2, 'r')
    contents = file.read()
    file.close
    parameter = []
    lenght = []
    width = []
    height = []
    weight = []
    for line in contents.split('\n'):
        row = line.split(' ')
        parameter.append(row[0])
        lenght.append(float(row[1]))
        width.append(float(row[2]))
        height.append(float(row[3]))
        weight.append(float(row[4]))
    finished_vehicle_data.append(parameter)
    finished_vehicle_data.append(lenght)
    finished_vehicle_data.append(width)
    finished_vehicle_data.append(height)
    finished_vehicle_data.append(weight)

    customer_orders_information_with_time_windows = []  # Customer orders information with time windowss
    file = open(archivo_3, 'r')
    contents = file.read()
    file.close
    labels = []
    latitude = []
    longitude = []
    earliest_time = []
    latest_time = []
    orders = []

    for line in contents.split('\n'):
        if len(labels) == numero_clientes+1:
            break
        row = line.split(' ')
        labels.append(row[0])
        latitude.append(float(row[1]))
        longitude.append(float(row[2]))
        if row[3] == '-':
            row[3] = '-'
        else:
            row[3] = int(row[3])
        earliest_time.append(row[3])
        if row[4] == '-':
            row[4] = '-'
        else:
            row[4] = int(row[4])
        latest_time.append(row[4])
        orders.append(row[5])


    customer_orders_information_with_time_windows.append(labels)
    customer_orders_information_with_time_windows.append(latitude)
    customer_orders_information_with_time_windows.append(longitude)
    customer_orders_information_with_time_windows.append(earliest_time)
    customer_orders_information_with_time_windows.append(latest_time)
    customer_orders_information_with_time_windows.append(orders_splitting(orders, finished_vehicle_data))

    # Parameter used
    file = open(archivo_4, 'r')
    contents = file.read()
    file.close
    kmton_cost = 0
    speed = 0
    oil_price = 0
    emission_index = 0
    road_condition = 0
    fuel_CEmission = 0
    carbon_tax = 0
    service_time = 0
    time_penalty = 0
    weight_factor = 0
    for line in contents.split('\n'):
        row = line.split(' ')
        kmton_cost = float(row[0])
        speed = int(row[1])
        oil_price = float(row[2])
        emission_index = float(row[3])
        road_condition = float(row[4])
        fuel_CEmission = float(row[5])
        carbon_tax = float(row[6])
        service_time = int(row[7])
        time_penalty = int(row[8])
        weight_factor = int(row[9])
    parameters.append(kmton_cost)
    parameters.append(speed)
    parameters.append(oil_price)
    parameters.append(emission_index)
    parameters.append(road_condition)
    parameters.append(fuel_CEmission)
    parameters.append(carbon_tax)
    parameters.append(service_time)
    parameters.append(time_penalty)
    parameters.append(weight_factor)
    
    # Data Poblation
    file = open(archivo_5, 'r')
    contents = file.read()
    file.close
    popsize = 0
    cross_prob = 0
    muta_prob = 0
    gen_max = 0
    for line in contents.split('\n'):
        row = line.split(' ')
        popsize = int(row[0])
        cross_prob = float(row[1])
        muta_prob = float(row[2])
        gen_max = int(row[3])
    poblation_data.append(popsize)
    poblation_data.append(cross_prob)
    poblation_data.append(muta_prob)
    poblation_data.append(gen_max)
    return trasport_vehicle_data, finished_vehicle_data,customer_orders_information_with_time_windows, parameters, poblation_data

#Haversine

def haversine(lat1, lon1, lat2, lon2):
    # Radio de la Tierra en kilómetros
    R = 6371.0
    # Convierte las coordenadas de grados a radianes
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)
    # Diferencia de latitudes y longitudes
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    # Fórmula de Haversine
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    # Distancia en kilómetros
    distance = R * c
    return distance

def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))

def coord(datos_clientes):
    x_y = []
    for i in range(1, len(datos_clientes[0])):
        x_y.append([datos_clientes[1][i] - datos_clientes[1][0], datos_clientes[2][i] - datos_clientes[2][0]])
    return x_y

def decision_x(pares_0, orden_vehiculos):
    pares = []
    if len(pares_0) == len(orden_vehiculos):
        for i in range(len(pares_0)):
            pares.append([pares_0[i], orden_vehiculos[i]])
    tipos = len(t_v_d[0])  # tipos = len(t_v_d[0])
    x = []
    uso = [0] * tipos
    while len(x) < len(orden_vehiculos):
        x.append(uso.copy())
    for i in range(tipos):
        for j in range(len(pares)):
            if pares[j][1] == i + 1:
                x[j][i] = 1
    return x

def decision_y(x, entre_0):
    tipos = len(t_v_d[0])  # tipos = len(t_v_d[0])
    y = []
    visitas = [0] * tipos
    while len(y) < len(c_o_i_t_w[0]) - 1:
        y.append(visitas.copy())
    for i in range(len(x)):
        for j in range(len(x[i])):
            if x[i][j] == 1:
                for cliente in entre_0[i]:
                    y[cliente - 1][j] = 1
    return y

def N_1(datos_clientes,datos_poblacion):
    i = 1
    n1 = []
    dat = []
    temp = []
    coords = coord(datos_clientes)

    for i in range(len(coords)):
        data = [angle_between([0, 0], coords[i]), i + 1]
        n1.append(data)
        i += 1
    shuffle(n1)

    for i in range(len(n1)):
        if n1[i][0] - n1[0][0] < 0:
            temp.append([n1[i][0] - n1[0][0] + 360, n1[i][1]])
        else:
            temp.append([n1[i][0] - n1[0][0], n1[i][1]])
    temp.sort()

    for i in range(len(temp)):
        temp[i] = temp[i][1]

    while len(dat) < len(n1):
        head = temp.pop(0)
        temp.append(head)
        temp_2 = temp.copy()
        dat.append(temp_2)

    temp = []
    for i in range(1, len(datos_clientes[0])):
        temp.append(i)
    while len(dat) < datos_poblacion[0]:
        temp = temp.copy()
        shuffle(temp)
        dat.append(temp)

    return dat

def N_2(datos_transportes):
    dat = []
    for i in range(len(datos_transportes[0])):
        if datos_transportes[1][i] > 0:
            dat += [datos_transportes[0][i]] * datos_transportes[1][i]
    shuffle(dat)
    return dat


def mejora(n1_original, n1_modificado, n2_original):
    camion_saltado = []
    camion_usado = []
    clientes = 0
    camiones = 0
    n1_mejorado = []
    n2_mejorado = []
    for i in range(len(n1_modificado)):
        if clientes == len(n1_original) and len(n1_mejorado) < len(n1_modificado):
            n1_mejorado.append(0)
            break
        elif clientes < len(n1_original) and i == len(n1_modificado):
            n1_modificado.extend(n1_original[clientes:])
            break
        if n1_modificado[i] != 0:
            n1_mejorado.append(n1_original[clientes])
            clientes += 1
        else:
            if i == 0:
                if n1_modificado[0] == 0 and n1_modificado[1] == 0:
                    camion_saltado.append(0)
                    camiones += 1
                else:
                    n1_mejorado.append(0)
                    camion_usado.append(0)
                    camiones += 1
            else:
                if n1_modificado[i] == 0 and n1_modificado[i+1] == 0:
                    camion_saltado.append(camiones)
                    camiones+=1
                else:
                    if camiones < len(n2_original):
                        n1_mejorado.append(0)
                        camion_usado.append(camiones)
                        camiones+=1
    for i in camion_usado:
        n2_mejorado.append(n2_original[i])
    n2_mejorado_extra = []
    for i in camion_saltado:
        n2_mejorado_extra.append(n2_original[i])
    return n1_mejorado,n2_mejorado

#Decodificador para visitas y regresos de vehiculos entre cantidad de clientes
def decodificar_pedidos(sin_modificar, n2_inicial, transportes, clientes):
    #Lista inicial para almacenar recorrido
    decodificado = [0]
    #Iterar sobre los vehiculos indicados en N2
    for vehicles in n2_inicial:
        temp_l = 0
        temp_w = 0
        #Iterar sobre los clientes en la lista sin modificar
        for cliente in sin_modificar:
            #Si el cliente ya pertenece al recorrdio se pasa al siguiente
            if cliente in decodificado:
                continue
            #Verificar si agregar al cliente excede las capacidades del vehiculo
            if temp_l + clientes[5][cliente][0] > transportes[2][vehicles - 1] or temp_w + clientes[5][cliente][1] > \
                    transportes[5][vehicles - 1]:
                #Si excede la capacidadse regresa al deposito y se rompe el ciclo
                decodificado.append(0)
                break
            #Actualizar cargas acumuladas con cliente actual
            temp_l += clientes[5][cliente][0]
            temp_w += clientes[5][cliente][1]
            decodificado.append(cliente)
            #Al llegar al ultimo cliente se agrega el deposito y se sale del ciclo
            if cliente == sin_modificar[-1]:
                decodificado.append(0)
                break
    #Verificacion de visitas a clientes
    if sin_modificar[-1] in decodificado:
        #Contador de cantidad de ceros en el recorrido
        ceros = 0
        for i in decodificado:
            if i == 0:
                ceros += 1
        return decodificado + [0] * (len(n2_inicial) - ceros + 1)
    else:
        for i in sin_modificar:
            if i not in decodificado:
                decodificado.append(i)
        return decodificado

#f_1(orden_v, pesos, distancias_por_porcion, entre_0)
def f_1(orden_vehiculos, pesos_por_trayecto, distancias_por_trayecto, entre_0,parametros,datos_v):
    vehicle_type = 0.5 * 0.7 * 0.005 *1.2041 #0.005 es el area del camion al frente en m**2 y 1.2041 es la densidad del aire en kg/m**3
    pesos = copy.deepcopy(pesos_por_trayecto)
    trayectos = 0
    var_x = [[],[],[]]
    for trayecto in entre_0:
        if len(trayecto) > 0:
            trayectos += 1
    for i in range(len(orden_vehiculos)):
        if orden_vehiculos[i] == 1:
            var_x[0].append(1)
            var_x[1].append(0)
            var_x[2].append(0)
        elif orden_vehiculos[i] == 2:
            var_x[0].append(0)
            var_x[1].append(1)
            var_x[2].append(0)
        else:
            var_x[0].append(0)
            var_x[1].append(0)
            var_x[2].append(1)
    total = 0
    costo_kilometro = parametros[0]
    oil_price = parametros[2]
    road_condition_factor = parametros[4]
    emission_index_parameter = parametros[3]
    speed = parametros[1]
    for trayecto in range(trayectos):
        inicio = 0
        mitad = 0
        final = 0
        inicio = inicio + var_x[0][trayecto] * datos_v[6][orden_vehiculos[trayecto] - 1]
        inicio = inicio + var_x[1][trayecto] * datos_v[6][orden_vehiculos[trayecto] - 1]
        inicio = inicio + var_x[2][trayecto] * datos_v[6][orden_vehiculos[trayecto] - 1]
        for porcion in range(len(pesos[trayecto])):
            peso_vehiculo = datos_v[7][orden_vehiculos[trayecto] - 1]
            peso_carga = pesos[trayecto][porcion]
            distancia_trayecto = distancias_por_trayecto[trayecto][porcion]
            mitad += var_x[0][trayecto] * costo_kilometro * ( peso_vehiculo + peso_carga) * distancia_trayecto
            mitad += var_x[1][trayecto] * costo_kilometro * ( peso_vehiculo + peso_carga) * distancia_trayecto
            mitad += var_x[2][trayecto] * costo_kilometro * ( peso_vehiculo + peso_carga) * distancia_trayecto
            final += oil_price * (var_x[0][trayecto] * emission_index_parameter * (road_condition_factor * (peso_vehiculo + peso_carga) + vehicle_type * (speed ** 2)) * distancia_trayecto)
            final += oil_price * (var_x[1][trayecto] * emission_index_parameter * (road_condition_factor * (peso_vehiculo + peso_carga) + vehicle_type * (speed ** 2)) * distancia_trayecto)
            final += oil_price * (var_x[2][trayecto] * emission_index_parameter * (road_condition_factor * (peso_vehiculo + peso_carga) + vehicle_type * (speed ** 2)) * distancia_trayecto)
        total += inicio + mitad + final
    return total
#f_2(lista_sin_modificar, tiempos)
def f_2(lista_sin_modificar, tiempos, data_clientes, parametros):
    early = 0
    late = 0
    for i in range(len(tiempos)):
        early += max([data_clientes[3][lista_sin_modificar[i]] - tiempos[i], 0])
        late += max([tiempos[i] - data_clientes[4][lista_sin_modificar[i]], 0])
    return parametros[8] * early + parametros[8] * late
#f_3(orden_v, pesos, distancias)
def f_3(orden_vehiculos, pesos_por_trayecto, distancias_por_trayecto, entre_0, parametros, datos_trasportes):
    vehicle_type = 0.5 * 0.7 * 0.005 *1.2041 #0.005 es el area del camion al frente en m**2 y 1.2041 es la densidad del aire en kg/m**3
    pesos = copy.deepcopy(pesos_por_trayecto)
    var_x = [[], [], []]
    costo = 0
    suma = 0
    conversion_factor = parametros[5]
    carbon_tax = parametros[6]
    road_factor = parametros[4]
    emission_index = parametros[3]
    speed = parametros[1]

    trayectos = 0
    var_x = [[],[],[]]
    for trayecto in entre_0:
        if len(trayecto) > 0:
            trayectos += 1
    for i in range(len(orden_vehiculos)):
        if orden_vehiculos[i] == 1:
            var_x[0].append(1)
            var_x[1].append(0)
            var_x[2].append(0)
        elif orden_vehiculos[i] == 2:
            var_x[0].append(0)
            var_x[1].append(1)
            var_x[2].append(0)
        else:
            var_x[0].append(0)
            var_x[1].append(0)
            var_x[2].append(1)
    
    for trayecto in range(trayectos):
        suma = 0
        for porcion in range(len(pesos[trayecto])):
            peso_vehiculo = datos_trasportes[7][orden_vehiculos[trayecto] - 1]
            peso_carga = pesos[trayecto][porcion]
            distancia_trayecto = distancias_por_trayecto[trayecto][porcion]
            suma += carbon_tax * (conversion_factor * (var_x[0][trayecto] * emission_index * (road_factor * (peso_vehiculo + peso_carga) + vehicle_type * (speed ** 2)) * distancia_trayecto))
            suma += carbon_tax * (conversion_factor * (var_x[1][trayecto] * emission_index * (road_factor * (peso_vehiculo + peso_carga) + vehicle_type * (speed ** 2)) * distancia_trayecto))
            suma += carbon_tax * (conversion_factor * (var_x[2][trayecto] * emission_index * (road_factor * (peso_vehiculo + peso_carga) + vehicle_type * (speed ** 2)) * distancia_trayecto))
        costo += suma
    return costo

def fitness_total(toda_la_info):
    total = 0
    max = -1
    minimo = 99999999999
    for i in toda_la_info:
        if i > max:
            max = i
        if i < minimo:
            minimo = i
        total += i
    return [minimo, total / len(toda_la_info), max]

def cross_prob(fitness, f_avg, f_max, data_pop):
    if fitness >= f_avg:
        return data_pop[1]
    return data_pop[1] * (f_max - f_avg) / (f_max - fitness)

def mut_prob(fitness, f_avg, f_max, data_pop):
    if fitness >= f_avg:
        return data_pop[2]
    return data_pop[2] * (f_max - f_avg) / (f_max - fitness)

#Funciones de Probabilidad

def seleccion(p_codificada: list, n_poblacion: int):
    #datos_esenciales = [N1,N1_MOD,N1_MEJ,N2_MEJ,FITNESS_ACTUAL,CROSS,MUT]
    seleccionados = []
    temp = copy.deepcopy(p_codificada)
    temporales = [[]]*len(temp[4])
    for i in range(len(temporales)):
        temporales[i] = [[temp[0][i],temp[1][i],temp[2][i],temp[3][i],temp[4][i],temp[5][i],temp[6][i]],i]
    temporales = sorted(temporales, key=lambda x: x[0][4],reverse=True)[:int(n_poblacion * 0.05)]
    for i in temporales:
        seleccionados.append(i[1])
    return seleccionados
    

def crossover(n2_sin_modificar: list, seleccion: list,data_fitness: list,poblacion: list, datos_pop: list, datos_clientes: list, datos_transportes: list, parametros: list):
    #poblacion = [lista_sin_modificar, fitness, cross, mut
    #poblacion_base_codificada[i] = [poblacion_base_codificada[i][0], poblacion_base_codificada[i][2], poblacion_base_codificada[i][3], poblacion_base_codificada[i][4]]
    #data = [poblacion_base_codificada, fitness_inicial, vehicle_order]
    #data[1],data[0] = crossover(selected,data[1],data[0])
    #data[1],data[0] = crossover(selected,fitness_actual[1],data[0])
    #datos_esenciales = [N1,N1_MOD,N1_MEJ,N2_MEJ,FITNESS_ACTUAL,cross,mut] print
    to_cross = copy.deepcopy(seleccion)
    nueva_poblacion = copy.deepcopy(poblacion)
    suma_cross = 0
    for i in poblacion[5]:
        suma_cross += i
    while len(to_cross) < datos_pop[0]/2:
        pivote = random() * suma_cross
        prob_acumulada = 0
        candidato = 0
        for i in range(len(poblacion[5])):
            prob_acumulada += poblacion[5][i]
            if prob_acumulada > pivote:
                candidato = i
                break
        if candidato not in to_cross:
            to_cross.append(candidato)  
    if len(to_cross) % 2 == 1:
        to_cross.pop()
    to_cross = [to_cross[:int(len(to_cross) / 2)], to_cross[int(len(to_cross) / 2):]]
    shuffle(to_cross[0])
    shuffle(to_cross[1])
    for i in range(len(to_cross[0])):
        crossing = pmx_crossover(poblacion[0][to_cross[0][i]], poblacion[0][to_cross[1][i]])
        nueva_poblacion[0][to_cross[0][i]] = crossing[0]
        nueva_poblacion[0][to_cross[1][i]] = crossing[1]
    to_cross = to_cross[0] + to_cross[1]
    for i in to_cross:
        temp_mod = decodificar_pedidos(nueva_poblacion[0][i],n2_sin_modificar,datos_transportes, datos_clientes)
        temp_n1_mejorado, temp_n2_mejorado = mejora(nueva_poblacion[0][i],temp_mod, n2_sin_modificar)
        fitness_temp = codificado_real(nueva_poblacion[0][i],temp_n1_mejorado,temp_n2_mejorado,datos_clientes,parametros, datos_transportes)
        nueva_poblacion[1][i] = temp_mod
        nueva_poblacion[2][i] = temp_n1_mejorado
        nueva_poblacion[3][i] = temp_n2_mejorado
        nueva_poblacion[4][i] = fitness_temp
    #Data Fitness
    fitness_crossover = fitness_total(nueva_poblacion[4])
    prob_cross_clientes = []
    prob_mut_clientes = []
    for i in range(len(nueva_poblacion[4])):
        prob_cross_clientes.append(cross_prob(nueva_poblacion[4][i], data_fitness[1], data_fitness[2],datos_pop))
        prob_mut_clientes.append(mut_prob(nueva_poblacion[4][i], data_fitness[1], data_fitness[2],datos_pop))
    nueva_poblacion.pop()
    nueva_poblacion.pop()
    nueva_poblacion.append(prob_cross_clientes)
    nueva_poblacion.append(prob_mut_clientes)
    return fitness_crossover,nueva_poblacion

def mutacion(data_fitness: list, poblacion: list, datos_pop: list, datos_transportes: list, datos_clientes: list, n2_sin_modificar: list, parametros):
    nueva_poblacion = copy.deepcopy(poblacion)
    nuevo_fitness = copy.deepcopy(data_fitness)
    mutated = []
    while len(mutated) < datos_pop[0]/2:
        suma_mut = 0
        for i in poblacion[6]:
            suma_mut += i
        pivote = random() * suma_mut
        prob_acumulada = 0
        candidato = 0
        for i in range(len(poblacion[0])):
            prob_acumulada += poblacion[6][i]
            if prob_acumulada > pivote:
                candidato = i
                break
        if candidato in mutated:
            continue
        mutated.append(candidato)
        r1 = randint(0, len(poblacion[0][0]))
        r2 = randint(0, len(poblacion[0][0]))
        while r1 >= r2:
            r1 = randint(0, len(poblacion[0][0]))
            r2 = randint(0, len(poblacion[0][0]))
        nueva_poblacion[0][candidato] = nueva_poblacion[0][candidato][:r1] + list(reversed(nueva_poblacion[0][candidato][r1:r2])) + nueva_poblacion[0][candidato][r2:]
        
        temp_mod = decodificar_pedidos(nueva_poblacion[0][candidato],n2_sin_modificar,datos_transportes, datos_clientes)
        temp_n1_mejorado, temp_n2_mejorado = mejora(nueva_poblacion[0][candidato],temp_mod, n2_sin_modificar)
        fitness_temp = codificado_real(nueva_poblacion[0][candidato],temp_n1_mejorado,temp_n2_mejorado,datos_clientes,parametros, datos_transportes)
        nueva_poblacion[1][candidato] = temp_mod
        nueva_poblacion[2][candidato] = temp_n1_mejorado
        nueva_poblacion[3][candidato] = temp_n2_mejorado
        nueva_poblacion[4][candidato] = fitness_temp
        
        #Data Fitness
        nuevo_fitness = fitness_total(nueva_poblacion[4])
        prob_cross_clientes = []
        prob_mut_clientes = []
        for i in range(len(nueva_poblacion[4])):
            prob_cross_clientes.append(cross_prob(nueva_poblacion[4][i], nuevo_fitness[1], nuevo_fitness[2],datos_pop))
            prob_mut_clientes.append(mut_prob(nueva_poblacion[4][i], nuevo_fitness[1], nuevo_fitness[2],datos_pop))
        nueva_poblacion.pop()
        nueva_poblacion.pop()
        nueva_poblacion.append(prob_cross_clientes)
        nueva_poblacion.append(prob_mut_clientes)
    return data_fitness,nueva_poblacion

def local_search_operator(data_fitness: list, poblacion: list, datos_pop: list, datos_transportes: list, datos_clientes: list, n2_sin_modificar: list, parametros):
    pop_lso = copy.deepcopy(poblacion)
    data_fitness_lso = copy.deepcopy(data_fitness)
    for i in range(len(pop_lso[0])):
        lsp = randint(0, len(pop_lso[0][i])-2)
        pair = [pop_lso[0][i][lsp],pop_lso[0][i][lsp+1]]
        candidate = copy.deepcopy(pop_lso[0][i])
        candidate[lsp] = pair[1]
        candidate[lsp+1] = pair[0]
        candidate_mod = decodificar_pedidos(candidate,n2_sin_modificar,datos_transportes, datos_clientes)
        candidate_n1_mejorado, candidate_n2_mejorado = mejora(candidate,candidate_mod, n2_sin_modificar)
        fitness_candidate = codificado_real(candidate,candidate_n1_mejorado,candidate_n2_mejorado,datos_clientes,parametros, datos_transportes)
        if fitness_candidate < pop_lso[4][i]:
            pop_lso[0][i] = candidate
            pop_lso[1][i] = candidate_mod
            pop_lso[2][i] = candidate_n1_mejorado
            pop_lso[3][i] = candidate_n2_mejorado
            pop_lso[4][i] = fitness_candidate
            data_fitness_lso = fitness_total(pop_lso[4])
            pop_lso[5][i] = cross_prob(pop_lso[4][i], data_fitness_lso[1], data_fitness_lso[2],datos_pop)
            pop_lso[6][i] = mut_prob(pop_lso[4][i], data_fitness_lso[1], data_fitness_lso[2],datos_pop)
        else:
            continue
    return pop_lso, data_fitness_lso

def pmx_crossover(parent1, parent2):
    # original code https://observablehq.com/@swissmanu/pmx-crossover
    """
    Performs Partially Mapped Crossover (PMX) on two parent sequences.
    Args:
        parent1: The first parent sequence.
        parent2: The second parent sequence.
    Returns:
        A tuple containing two child sequences (offspring).
    Raises:
        ValueError: If the parents have different lengths.
    """

    if len(parent1) != len(parent2):
        raise ValueError("Parents must have the same length!")

    # Select random crossover points
    crossover_point1 = randint(0, len(parent1) - 2)
    crossover_point2 = randint(crossover_point1 + 1, len(parent1))

    # Create child sequences (shallow copies)
    child1 = parent1.copy()
    child2 = parent2.copy()

    # Mapping for repairs
    mapping1 = {parent2[i]: parent1[i] for i in range(crossover_point1, crossover_point2)}
    mapping2 = {parent1[i]: parent2[i] for i in range(crossover_point1, crossover_point2)}

    # Swap elements within the slice
    for i in range(crossover_point1, crossover_point2):
        child1[i] = parent2[i]
        child2[i] = parent1[i]

    # Repair lower and upper slices
    for i in range(crossover_point1):
        while child1[i] in mapping1:
            child1[i] = mapping1[child1[i]]
        while child2[i] in mapping2:
            child2[i] = mapping2[child2[i]]

    for i in range(crossover_point2, len(parent1)):
        while child1[i] in mapping1:
            child1[i] = mapping1[child1[i]]
        while child2[i] in mapping2:
            child2[i] = mapping2[child2[i]]
    child1 = [child1[:crossover_point1], child1[crossover_point1:crossover_point2], child1[crossover_point2:]]
    child1[1].reverse()
    child2 = [child2[:crossover_point1], child2[crossover_point1:crossover_point2], child2[crossover_point2:]]
    child2[1].reverse()
    return [child1[0] + child1[1] + child1[2], child2[0] + child2[1] + child2[2]]


def codificado_real( lista_inicial, lista_mejorada, orden_v_mej, info_clientes, parametros, data_v):
    lat = info_clientes[1]
    long = info_clientes[2]
    pesos = []
    tiempos = []
    pares_0 = []
    entre_0 = []
    distancias = []
    posicion_ceros = []
    vehiculo = 0
    for i in range(len(lista_mejorada)):
        if lista_mejorada[i] == 0:
            posicion_ceros.append(i)
    for i in range(len(posicion_ceros) - 1):
        sublista = [posicion_ceros[i], posicion_ceros[i + 1]]
        pares_0.append(sublista)
    for par in pares_0:
        distancia = []
        aux = []
        w = []
        tiempo = []
        for j in range(par[0] + 1, par[1]):
            aux.append(lista_mejorada[j])
        for j in aux:
            w.append(info_clientes[5][j][1])
        for j in range(len(w)):
            w[j] = sum(w[j:])
        if len(aux) > 0:
            j = 0
            while j < len(aux):
                if j == 0:
                    distancia.append(haversine(lat[0], long[0], lat[aux[j]], long[aux[j]]))
                if 0 <= j < len(aux) - 1:
                    distancia.append(haversine(lat[aux[j + 1]], long[aux[j + 1]], lat[aux[j]], long[aux[j]]))
                if j == len(aux) - 1:
                    distancia.append(haversine(lat[0], long[0], lat[aux[j]], long[aux[j]]))
                j += 1
        for i in distancia:
            tiempo.append(i / parametros[1])

        tiempos.append(tiempo)
        distancias.append(distancia)
        entre_0.append(aux)
        if len(w) > 0:
            w.append(0)
        pesos.append(w)
        vehiculo += 1
    resto = 0
    temp = []
    for i in tiempos:
        if i == []:
            break
        while len(i) > 1:
            if resto > 0:
                temp.append(i.pop(0) + resto)
                resto = 0
            else:
                temp.append(i.pop(0))
        resto = i.pop(0)
    tiempos = copy.deepcopy(temp)
    temp = []
    resto = 0
    distancias_por_porcion = copy.deepcopy(distancias)
    for i in distancias:
        if i == []:
            break
        while len(i) > 1:
            if resto > 0:
                temp.append(i.pop(0) + resto)
                resto = 0
            else:
                temp.append(i.pop(0))
        resto = i.pop(0)
    distancias = copy.deepcopy(temp)
    fitness = f_1(orden_v_mej, pesos, distancias_por_porcion, entre_0, parametros, data_v)
    fitness += f_2(lista_inicial, tiempos, info_clientes,parametros)
    fitness += f_3(orden_v_mej, pesos, distancias_por_porcion, entre_0,parametros,data_v)  #print(f_1(orden_v, pesos, distancias_por_porcion, entre_0,p_o_m,t_v_d),f_2(lista_inicial, tiempos),f_3(orden_v, pesos, distancias_por_porcion, entre_0))
    #lista_codificada = [lista_sin_modificar_inicial, lista_pre_codificada_inicial, fitness, 0, 0, pares_0, entre_0, pesos, distancias, tiempos]

    return fitness

def hvrp_fvl(file_1, file_2, file_3, file_4, file_5, folder_data, n_clientes):
    #HVRP-FVL():
    #Paso 1
    #Leer datos
    t_v_d, f_v_i, c_o_i_t_w, p_o_m, pop_data= lectura_archivos(file_1, file_2, file_3, file_4, file_5, folder_data,n_clientes)
    #Paso 2
    #Crear N1 y N2
    poblacion_inicial = N_1(c_o_i_t_w,pop_data)
    vehicle_order_inicial = N_2(t_v_d)
    datos_esenciales = []
    datos_esenciales.append(poblacion_inicial)
    #Paso 3
    #Modificar N1
    poblacion_inicial_modificada = []
    for poblacion in poblacion_inicial:
        poblacion_inicial_modificada.append(decodificar_pedidos(poblacion, vehicle_order_inicial, t_v_d, c_o_i_t_w))
    datos_esenciales.append(poblacion_inicial_modificada)
    #Mejora N1 modificado / N2 mejorado
    poblacion_inicial_mejorada = []
    vehicle_order_mejorado = []
    for i in range(len(poblacion_inicial)):
        pop_mejorada, orden_mejorado = mejora(poblacion_inicial[i],poblacion_inicial_modificada[i],vehicle_order_inicial)
        poblacion_inicial_mejorada.append(pop_mejorada)
        vehicle_order_mejorado.append(orden_mejorado)
    datos_esenciales.append(poblacion_inicial_mejorada)
    datos_esenciales.append(vehicle_order_mejorado)

    #datos_esenciales = [N1,N1_MOD,N1_MEJ,N2_MEJ,FITNESS_ACTUAL,CROSS,MUT]
    #Calculo Fitness
    fitness_actual = []
    for i in range(len(datos_esenciales[2])):
        fitness_actual.append(codificado_real(datos_esenciales[0][i],datos_esenciales[2][i],datos_esenciales[3][i],c_o_i_t_w, p_o_m, t_v_d))
    datos_esenciales.append(fitness_actual)
    fitness_generacion = fitness_total(datos_esenciales[4])
    prob_cross_clientes = []
    prob_mut_clientes = []
    for i in range(len(datos_esenciales[4])):
        prob_cross_clientes.append(cross_prob(datos_esenciales[4][i], fitness_generacion[1], fitness_generacion[2],pop_data))
        prob_mut_clientes.append(mut_prob(datos_esenciales[4][i], fitness_generacion[1], fitness_generacion[2],pop_data))
    datos_esenciales.append(prob_cross_clientes)
    datos_esenciales.append(prob_mut_clientes)

    minimo = 99999999999999
    pos_minimo = 0
    for i in range(len(datos_esenciales[4])):
        if datos_esenciales[4][i] < minimo:
            minimo = datos_esenciales[4][i]
            pos_minimo = i
    mejores_candidatos_por_generacion = [[datos_esenciales[2][pos_minimo],datos_esenciales[3][pos_minimo],minimo]]# [n1_mej,n2_mej,fitness_opcion]
    print("-------------------------------------Inicio GA--------------------------------")
    tiempos = [["-","-","-"]]
    t_i = 0
    t_f = 0

    gen = 0
    while gen < pop_data[3]:
        tiempo_generacion = [0,0,0] #Crossover,Mutation,LSO
        print("---------------------------------Generacion",str(gen+1)+"--------------------------------")
        selected = seleccion(datos_esenciales,pop_data[0])
        t_i = time.time()
        fitness_generacion,datos_esenciales = crossover(vehicle_order_inicial, selected, fitness_generacion, datos_esenciales, pop_data, c_o_i_t_w, t_v_d, p_o_m)
        t_f = time.time()
        tiempo_generacion[0] = t_f-t_i
        t_i = time.time()
        fitness_generacion,datos_esenciales = mutacion(fitness_generacion,datos_esenciales,pop_data,t_v_d,c_o_i_t_w,vehicle_order_inicial,p_o_m)
        t_f = time.time()
        tiempo_generacion[1] = t_f-t_i
        t_i = time.time()
        datos_esenciales, fitness_generacion = local_search_operator(fitness_generacion,datos_esenciales,pop_data,t_v_d,c_o_i_t_w,vehicle_order_inicial,p_o_m)
        t_f = time.time()
        tiempo_generacion[2] = t_f-t_i
        

        minimo = 99999999999999
        pos_minimo = 0
        for i in range(len(datos_esenciales[5])):
            if datos_esenciales[4][i] < minimo:
                minimo = datos_esenciales[4][i]
                pos_minimo = i
        mejores_candidatos_por_generacion.append([datos_esenciales[2][pos_minimo],datos_esenciales[3][pos_minimo],minimo])# [n1_mej,n2_mej,fitness_opcion]
        #se ve como quedan los datos y se deja con el formato para el crossover
        gen += 1
        tiempos.append(tiempo_generacion)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    carpeta_resultados = os.path.join(current_dir, 'resultados_pruebas')
    resultados = os.path.join(carpeta_resultados, f"resultado_especial_{n_clientes}_clientes_maximo_{file_3[0]}_por_tipo_maximo_{file_3[1]}_por_cliente.txt")
    f = open(resultados, "w")
    f.write(f"Generacion,tiempo_crossover,tiempo_mutation,tiempo_lso,N1,N2,Fitness\n")
    for i in range(len(mejores_candidatos_por_generacion)):
        if i == 0:
            f.write(f"G{i},-,-,-,{mejores_candidatos_por_generacion[i][0]},{mejores_candidatos_por_generacion[i][1]},{mejores_candidatos_por_generacion[i][2]}\n")
            continue
        if i == len(mejores_candidatos_por_generacion)-1:
            f.write(f"G{i},{tiempos[i][0]},{tiempos[i][1]},{tiempos[i][2]},{mejores_candidatos_por_generacion[i][0]},{mejores_candidatos_por_generacion[i][1]},{mejores_candidatos_por_generacion[i][2]}")
        else:
            f.write(f"G{i},{tiempos[i][0]},{tiempos[i][1]},{tiempos[i][2]},{mejores_candidatos_por_generacion[i][0]},{mejores_candidatos_por_generacion[i][1]},{mejores_candidatos_por_generacion[i][2]}\n")
    f.close
    print(f"Resultados guardados como resultado_especial_{n_clientes}_clientes_maximo_{file_3[0]}_por_tipo_maximo_{file_3[1]}_por_cliente")

numero_clientes = [10,20,30,40,50]
ordenes = [[1,5],[2,5],[2,10],[3,5],[3,10],[3,15],[4,5],[4,10],[4,15],[4,20],[5,5],[5,10],[5,15],[5,20],[5,25]]
tiempos = []
current_dir = os.path.dirname(os.path.abspath(__file__))
carpeta_resultados = os.path.join(current_dir, 'resultados_pruebas')
datos_y_tiempos = os.path.join(carpeta_resultados, f"resultados_y_tiempos2.txt")
f = open(datos_y_tiempos, "w")
f.write(f"n_clientes,maximo_por_tipo,maximo_por_cliente,tiempo_HVRP-FVL\n")
for cantidad in numero_clientes:
    for orden in ordenes:
        t_i = time.time()
        hvrp_fvl('vehicle_data1.txt','finish_vehicle_data.txt',orden, 'model_parameters.txt', 'pop_data.txt','data_pruebas', cantidad)
        t_f = time.time()
        tiempos.append(t_f-t_i)
i = 0
for cantidad in numero_clientes:
    for orden in ordenes:
        f.write(f"{cantidad},{orden[0]},{orden[1]},{tiempos[i]}\n")
        print(f"N° Clientes: {cantidad} | Maximo por tipo: {orden[0]} | Maximo por cliente: {orden[1]} | Tiempo HVRP-FVL: {tiempos[i]} segundos.")
        i += 1 
f.close()

"""numero_clientes = [60,70,80,90,100]
tiempos = []
current_dir = os.path.dirname(os.path.abspath(__file__))
carpeta_resultados = os.path.join(current_dir, 'resultados_pruebas')
datos_y_tiempos = os.path.join(carpeta_resultados, f"resultados_y_tiempos2.txt")
f = open(datos_y_tiempos, "w")
f.write(f"n_clientes maximo_por_tipo maximo_por_cliente tiempo_HVRP-FVL\n")
for cantidad in numero_clientes:
    for orden in ordenes:
        t_i = time.time()
        hvrp_fvl('vehicle_data1.txt','finish_vehicle_data.txt',orden, 'model_parameters.txt', 'pop_data.txt','data_pruebas', cantidad)
        t_f = time.time()
        tiempos.append(t_f-t_i)
i = 0
for cantidad in numero_clientes:
    for orden in ordenes:
        f.write(f" {cantidad} {orden[0]} {orden[1]} {tiempos[i]}\n")
        print(f"N° Clientes: {cantidad} | Maximo por tipo: {orden[0]} | Maximo por cliente: {orden[1]} | Tiempo HVRP-FVL: {tiempos[i]} segundos.")
        i += 1 
f.close()"""