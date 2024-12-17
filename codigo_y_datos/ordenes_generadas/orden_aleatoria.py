import random

# Función para generar una orden aleatoria, se ocupa el valor de los camiones de transporte
def generar_orden_aleatoria_por_tipos():
    v_t = [['A', 'B', 'C', 'D', 'E'],  # Tipos de carga
       [3.46, 4.135, 4.35, 4.515, 4.865],  # Longitud por tipo (m)
       [0.87, 1.225, 1.27, 1.32, 1.645]]  # Peso por tipo (t)

    largo_maximo = 67  # Largo máximo del camión en metros
    peso_maximo = 40   # Peso máximo del camión en toneladas
    while True:
        tipos_carga = v_t[0]
        largos = v_t[1]
        pesos = v_t[2]
        
        # Generar cantidades aleatorias para los 5 tipos
        orden = {}
        largo_total = 0
        peso_total = 0
        
        for tipo, largo, peso in zip(tipos_carga, largos, pesos):
            cantidad = random.randint(0, 5)  # Cantidad aleatoria entre 0 y 5
            largo_total += cantidad * largo
            peso_total += cantidad * peso
            if cantidad > 0:
                orden[tipo] = cantidad
        
        # Validar la orden completa
        if largo_total <= largo_maximo and peso_total <= peso_maximo:
            # Si es válida, retornar la orden
            return [[largo_total,peso_total],"+".join(f"{cantidad}{tipo}" for tipo, cantidad in sorted(orden.items()))]

# Uso
orden_generada = generar_orden_aleatoria_por_tipos()
print("Orden generada:", orden_generada[1])
print("Largo y Peso asociado:", orden_generada[0])

# Función para validar si una orden cabe en un camión
def validar_orden_en_camion(orden, camion):
    # Datos de ejemplo
    v_t = [['A', 'B', 'C', 'D', 'E'],  # Tipos de carga
        [3.46, 4.135, 4.35, 4.515, 4.865],  # Longitud por tipo (m)
        [0.87, 1.225, 1.27, 1.32, 1.645]]  # Peso por tipo (t)
    tipos_carga = v_t[0]
    largos = v_t[1]
    pesos = v_t[2]
    
    # Calcular el largo total y el peso total de la orden
    largo_total = sum(orden[tipo] * largos[tipos_carga.index(tipo)] for tipo in orden)
    peso_total = sum(orden[tipo] * pesos[tipos_carga.index(tipo)] for tipo in orden)
    
    # Validar si la orden cabe en el camión
    return largo_total <= camion["largo"] and peso_total <= camion["peso"]

def calcular_minimo_camiones_por_tipo(ordenes):
    """
    Calcula el número mínimo de cada tipo de camión necesario para transportar las órdenes.

    Args:
        ordenes (list): Lista de órdenes, cada una como [largo, peso].

    Returns:
        dict: Número de camiones necesarios por tipo.
    """
    # Capacidades de los camiones (peso en toneladas, largo en metros)
    camiones = [
        {"tipo": 1, "peso": 20, "largo": 38},  # Camión tipo 1 (el más pequeño)
        {"tipo": 2, "peso": 30, "largo": 52},  # Camión tipo 2
        {"tipo": 3, "peso": 40, "largo": 67}   # Camión tipo 3 (el más grande)
    ]

    camiones_necesarios = {camion["tipo"]: 0 for camion in camiones}

    for orden in ordenes:
        # Intentar asignar al camión más pequeño disponible
        for camion in camiones:
            if orden[0] <= camion["largo"] and orden[1] <= camion["peso"]:
                camiones_necesarios[camion["tipo"]] += 1
                break

    return camiones_necesarios

# Ejemplo de uso
ordenes = [[36.435, 10.305], [62.325, 18.69], [36.599999999999994, 10.355], [37.135, 10.955]]
resultado = calcular_minimo_camiones_por_tipo(ordenes)
print("Camiones necesarios por tipo:", resultado)