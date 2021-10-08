texto = '''
Un caballo viejo fue vendido para darle vueltas a la piedra de un molino.
Al verse atado a la piedra, exclamó sollozando: - ¡Después de las vueltas
de las carreras, he aquí a que vueltas me he reducido! Moraleja:
No presumáis de la fortaleza de la juventudhola. Para muchos,
la vejez es un trabajo muy penosoholahola AEROPLANO.
'''

por_encontrar = (
    'caballo', 'en', 'aeroplano', 'hola', 'corona'
)
resultado = []
for coincidencia in por_encontrar:
    if coincidencia in texto.lower():
        resultado.append(True)
    else:
        resultado.append(False)

print(f'se ecnotraron las siguientes palabras {resultado}')