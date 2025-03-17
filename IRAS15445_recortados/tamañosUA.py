import matplotlib.pyplot as plt
import numpy as np

# Definir el rango de distancias
d = np.linspace(3.5, 10, 100)

# Ángulo en radianes
ang = np.array([0.4,0.4]) / 3600 * np.pi / 180  # rad

plt.figure(figsize=(10, 5))

# Sombrear áreas de error
plt.fill_betweenx([100,5000], 4.38-0.69, 4.38+0.69, color='purple', alpha=0.1)
plt.fill_betweenx([100,5000], 5.4 - 0.4, 5.4 + 0.4, color='purple', alpha=0.1)
plt.fill_betweenx([100,5000], 8.5 - 0.4, 8.5 + 0.4, color='purple', alpha=0.1)

# Marcar líneas verticales en 4.4, 5.4 y 8.5 kpc
plt.axvline(4.38, color='purple', linestyle='--')
plt.axvline(5.4, color='purple', linestyle='--')
plt.axvline(8.5, color='purple', linestyle='--')

plt.text(4.45, 3300, '4.38 kpc', fontsize=11)
plt.text(5.45, 3300, '5.4 kpc', fontsize=11)
plt.text(8.55, 3300, '8.5 kpc', fontsize=11)

plt.xlim(3.5,9)
plt.ylim(1400,5000)

# Etiquetas
plt.title('SO2 (21-21)',fontsize=13)
plt.xlabel('Distancia [kpc]', fontsize=13)
plt.ylabel('Tamaño [UA]', fontsize=13)


a = ['Horizontal','Vertical']

for i in range(len(a)):
    # Cálculo del tamaño
    x = 2 * d * np.tan(ang/ 2) * 2.063e+8 # Factor kpc -> UA
    
    plt.plot(d, x, label=a[i])

plt.legend(loc='upper left',fontsize=11)
plt.grid()