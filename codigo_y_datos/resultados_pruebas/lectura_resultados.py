import os
import re
import pandas as pd

def lectura(numero_clientes,ordenes):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    lista_analisis = []
    n_archivo = 0
    for cantidad in numero_clientes:
        for orden in ordenes:
            lista_analisis.append([])
            resultado = os.path.join(current_dir, f"resultado_{cantidad}_clientes_maximo_{orden[0]}_por_tipo_maximo_{orden[1]}_por_cliente.txt")
            f = open(resultado, "r")
            datos = f.readlines()
            f.close()
            for linea in range(len(datos)):
                real = []
                if linea == 0 or linea == 1:
                    continue
                l = datos[linea].replace("\n","").replace("] [","|").replace("] ","|").replace(" [","|").split("|")
                l[0] = l[0].split()
                l[1] = re.findall(r'\d+', l[1])
                l[2] = re.findall(r'\d+', l[2])
                real.append(linea-1)
                real.append(float(l[0][1]))
                real.append(float(l[0][2]))
                real.append(float(l[0][3]))
                real.append([])
                real.append([])
                for i in range(len(l[1])):
                    real[4].append(int(l[1][i]))
                for i in range(len(l[2])):
                    real[5].append(int(l[2][i]))
                real.append(float(l[3]))
                lista_analisis[n_archivo].append(real)
            n_archivo += 1
        
        
    info = []
    for analisis in range(len(lista_analisis)):
        gen_max = -1
        lugar_gen_max = 0
        gen_min = 9999999999999999999999999999
        lugar_gen_min = 0
        gen_prom = 0
        for gen in range(1000):
            if lista_analisis[analisis][gen][6] > gen_max:
                gen_max = lista_analisis[analisis][gen][6]
                lugar_gen_max = gen
            if lista_analisis[analisis][gen][6] < gen_min:
                gen_min = lista_analisis[analisis][gen][6]
                lugar_gen_min = gen
            gen_prom += lista_analisis[analisis][gen][6]
        info.append([gen_max,lugar_gen_max,gen_min,lugar_gen_min,gen_prom/1000])

    resultado = 0
    for cantidad in numero_clientes:
        for orden in ordenes:
            info[resultado] = [cantidad,orden[0],orden[1]]+info[resultado]
            resultado += 1

    archivo_analisis = os.path.join(current_dir, f"analisis.txt")
    f = open(archivo_analisis, "w")
    f.write("numero_clientes,maximo_por_tipo,maximo_por_cliente,gen_max,lugar_gen_max,gen_min,lugar_gen_min,gen_prom\n")
    for i in range(len(info)):
        if info[i] == info[-1]:
            f.write(f"{info[i][0]},{info[i][1]},{info[i][2]},{info[i][3]},{info[i][4]},{info[i][5]},{info[i][6]},{info[i][7]}")
            continue
        f.write(f"{info[i][0]},{info[i][1]},{info[i][2]},{info[i][3]},{info[i][4]},{info[i][5]},{info[i][6]},{info[i][7]}\n")
    f.close()

    df = pd.read_csv(archivo_analisis, delimiter=',')
    excel = os.path.join(current_dir, 'analisis.xlsx')
    df.to_excel(excel, index=False)
    print("Data successfully converted and saved to output.xlsx")
    #return info

n = [10,20,30,40,50]
o = [[1,5],[2,5],[2,10],[3,5],[3,10],[3,15],[4,5],[4,10],[4,15],[4,20],[5,5],[5,10],[5,15],[5,20],[5,25]]

analisis = lectura(n,o)
"""print("numero_clientes,maximo_por_tipo,maximo_por_cliente,gen_max,lugar_gen_max,gen_min,lugar_gen_min,gen_prom")
for i in analisis:
    print(f"{i[0]},{i[1]},{i[2]},{i[3]},{i[4]},{i[5]},{i[6]},{i[7]}")
excel = "analisis.xlsx"""