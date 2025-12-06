# Red Neuronal ART

Este proyecto es una aplicación de escritorio desarrollada en Python con Tkinter que permite simular y visualizar el funcionamiento de una red neuronal de Teoría de Resonancia Adaptativa (ART). La aplicación permite entrenar la red con patrones personalizados o cargados desde imágenes, y observar cómo la red clasifica y aprende nuevas categorías.

## Características

*   **Interfaz Gráfica Interactiva**: Permite dibujar patrones en una cuadrícula de 28x28 o cargar imágenes.
*   **Visualización en Tiempo Real**: Muestra los patrones que la red ha aprendido y almacenado en sus neuronas.
*   **Parámetro de Vigilancia Ajustable**: Se puede modificar el parámetro de vigilancia (rho) para controlar la similitud requerida para agrupar patrones en una misma categoría.
*   **Entrenamiento y Prueba**: Funciones para entrenar la red con un nuevo patrón y para probar a qué categoría pertenece un patrón existente.
*   **Gestión de la Red**: Es posible resetear la red para borrar todos los patrones aprendidos y comenzar de nuevo.

## ¿Cómo funciona?

La aplicación utiliza una red neuronal ART para clasificar patrones binarios de 28x28 píxeles. El flujo de trabajo es el siguiente:

1.  **Creación de un Patrón**: El usuario puede crear un patrón de dos maneras:
    *   **Dibujando en la cuadrícula**: Haciendo clic en las celdas de la cuadrícula para activarlas o desactivarlas.
    *   **Cargando una imagen**: La aplicación carga una imagen, la convierte a escala de grises, la invierte, la redimensiona a 28x28 y la convierte en un patrón binario.
2.  **Entrenamiento de la Red**: Al presionar "Entrenar Red", el patrón actual se presenta a la red ART.
    *   La red calcula la similitud del patrón con los patrones previamente aprendidos (prototipos) en sus neuronas.
    *   Si el patrón es lo suficientemente similar a un prototipo existente (según el parámetro de vigilancia), la red actualiza ese prototipo para refinarlo.
    *   Si el patrón no es lo suficientemente similar a ningún prototipo existente, la red lo aprende como una nueva categoría, asignándolo a una nueva neurona.
3.  **Prueba de un Patrón**: Al presionar "Probar Patrón", la red determina a qué categoría (neurona) pertenece el patrón actual sin aprenderlo.
4.  **Visualización de Prototipos**: La sección "Patrones Aprendidos" muestra los prototipos almacenados en cada neurona comprometida.

## Archivos del Proyecto

*   `main.py`: Contiene el código principal de la aplicación, incluyendo la interfaz gráfica de usuario (GUI) construida con Tkinter y la lógica de interacción con el usuario.
*   `art_network.py`: Implementa la clase `ART` que contiene la lógica de la red neuronal de Teoría de Resonancia Adaptativa.
*   `image_processor.py`: Contiene la función para procesar imágenes (cargarlas, redimensionarlas y convertirlas en patrones binarios).
*   `requirements.txt`: Lista las dependencias de Python necesarias para ejecutar el proyecto.

## Instalación

1.  Clona este repositorio o descarga los archivos en tu máquina local.
2.  Abre una terminal en el directorio del proyecto.
3.  Crea un entorno virtual (recomendado):
    ```bash
    python -m venv venv
    ```
4.  Activa el entorno virtual:
    *   En Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    *   En macOS/Linux:
        ```bash
        source venv/bin/activate
        ```
5.  Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```

## Uso

Para ejecutar la aplicación, asegúrate de tener el entorno virtual activado y las dependencias instaladas. Luego, ejecuta el siguiente comando en la terminal:

```bash
python main.py
```

