import os
import re
import pandas as pd

def lectura(numero_clientes,ordenes,arh1,arh2):
    #0,1,2,3,lugar_gen_max,5,lugar_gen_min,7
    current_dir = os.path.dirname(os.path.abspath(__file__))
    carpeta_resultados_1 = os.path.join(current_dir, "resultados_experimento_1")
    carpeta_resultados_2 = os.path.join(current_dir, "resultados_experimento_2")
    analisis_resultados_1 = os.path.join(carpeta_resultados_1, "analisis_experimento_1.txt")
    analisis_resultados_2 = os.path.join(carpeta_resultados_2, "analisis_experimento_2.txt")
    lista_analisis = []
    f = open(analisis_resultados_1, "r")
    datos_analisis_1 = f.readlines()
    f.close()
    f = open(analisis_resultados_2, "r")
    datos_analisis_2 = f.readlines()
    f.close()
    lugares = [[],[]]
    for linea in range(len(datos_analisis_1)):
        if linea == 0:
            continue
        l = datos_analisis_1[linea].replace("\n","").split(",")
        lugares[0].append([int(l[6]),int(l[4])])
    for linea in range(len(datos_analisis_2)):
        if linea == 0:
            continue
        l = datos_analisis_2[linea].replace("\n","").split(",")
        lugares[1].append([int(l[6]),int(l[4])])
    #resultado_10_clientes_maximo_1_por_tipo_maximo_5_por_cliente.txt
    rutas = [[],[]]
    n_archivo = 0
    for cantidad in numero_clientes:
        for orden in ordenes:
            resultado = os.path.join(carpeta_resultados_1, f"resultado_{cantidad}_clientes_maximo_{orden[0]}_por_tipo_maximo_{orden[1]}_por_cliente.txt")
            f = open(resultado, "r")
            datos = f.readlines()
            f.close()
            datos = datos[2:]
            real = []
            menor = datos[lugares[0][n_archivo][0]].replace("\n","").replace("] [","|").replace("] ","|").replace(" [","|").split("|")
            menor[0] = menor[0].split()
            menor[1] = re.findall(r'\d+', menor[1])
            menor[2] = re.findall(r'\d+', menor[2])
            real.append(linea+1)
            real.append(float(menor[0][1]))
            real.append(float(menor[0][2]))
            real.append(float(menor[0][3]))
            real.append([])
            real.append([])
            for i in range(len(menor[1])):
                real[4].append(int(menor[1][i]))
            for i in range(len(menor[2])):
                real[5].append(int(menor[2][i]))
            real.append(float(menor[3]))
            real = []
            menor = datos[lugares[0][n_archivo][0]].replace("\n","").replace("] [","|").replace("] ","|").replace(" [","|").split("|")
            menor[0] = menor[0].split()
            menor[1] = re.findall(r'\d+', menor[1])
            menor[2] = re.findall(r'\d+', menor[2])
            real.append(linea+1)
            real.append(float(menor[0][1]))
            real.append(float(menor[0][2]))
            real.append(float(menor[0][3]))
            real.append([])
            real.append([])
            for i in range(len(menor[1])):
                real[4].append(int(menor[1][i]))
            for i in range(len(menor[2])):
                real[5].append(int(menor[2][i]))
            real.append(float(menor[3]))
            rutas[0].append(real)
            n_archivo += 1
    """archivo_analisis = os.path.join(current_dir, f"analisis_tiempos_juntos.txt")
    f = open(archivo_analisis, "w")
    f.write("n_clientes,maximo_por_tipo,maximo_por_cliente,tiempo_HVRP-FVL_prueba_1,tiempo_HVRP-FVL_prueba_2\n")
    for i in range(len(datos)):
        if datos[i] == datos[-1]:
            f.write(f"{datos[i][0]},{datos[i][1]},{datos[i][2]},{datos[i][3]},{datos[i][4]}")
            continue
        f.write(f"{datos[i][0]},{datos[i][1]},{datos[i][2]},{datos[i][3]},{datos[i][4]}\n")
    f.close()

    df = pd.read_csv(archivo_analisis, delimiter=',')
    excel = os.path.join(current_dir, 'analisis_tiempos.xlsx')
    df.to_excel(excel, index=False)
    print("Data successfully converted and saved to analisis_tiempos.xlsx")"""
    for i in rutas[0]:
        print(i)
    #return info

experimento_1 = "resultados_y_tiempos_experimento_1.txt"
experimento_2 = "resultados_y_tiempos_experimento_2.txt"
n = [10,20,30,40,50]
o = [[1,5],[2,5],[2,10],[3,5],[3,10],[3,15],[4,5],[4,10],[4,15],[4,20],[5,5],[5,10],[5,15],[5,20],[5,25]]

analisis = lectura(n, o, experimento_1, experimento_2)