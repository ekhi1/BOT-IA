import discord
import random
from discord.ext import commands

import requests
import os

from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

model1 = r"C:\Users\ekhir\Downloads\Programacion\Python\proyecto_de_seguridad_automobilistica\keras_model1.h5"
model2 = r"C:\Users\ekhir\Downloads\Programacion\Python\proyecto_de_seguridad_automobilistica\keras_model2.h5"
label1 = r"C:\Users\ekhir\Downloads\Programacion\Python\proyecto_de_seguridad_automobilistica\labels1.txt"
label2 = r"C:\Users\ekhir\Downloads\Programacion\Python\proyecto_de_seguridad_automobilistica\labels2.txt"

def detect_ia_on_road(model, labels, image):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model(model, compile=False)

    # Load the labels
    class_names = open(labels, "r").readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open(image).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    confidence_score = confidence_score * 100

    # Print prediction and confidence score
    result = []
    for i in range(len(class_names)):
        result.append(class_names[i][2:-1]+':'+str(np.round((100 * (np.round(prediction[0][i],3))), 4)))

    final_result = ["Class: "+ class_name[2:], "Confidence Score: "+str(np.round(confidence_score, 3))]

    return result, final_result

def deteccion_tecnologica(model, labels, image):

    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model(model, compile=False)

    # Load the labels
    class_names = open(labels, "r").readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open(image).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    confidence_score = confidence_score * 100


    result = []
    for i in range(len(class_names)):
        result.append(class_names[i][2:-1]+':'+str(np.round((100 * (np.round(prediction[0][i],3))), 4)))

    final_result = ["Class: "+ class_name[2:], "Confidence Score: "+str(np.round(confidence_score, 3))]

    return result, final_result

memes = ["proyect 1/Imagenes/meme_programacion.jpeg",
        "proyect 1/Imagenes/meme2.jpeg",
        "proyect 1/Imagenes/meme_bideojuegos1.jpg",
        "proyect 1/Imagenes/meme_bideojuegos2.jpg",
        "proyect 1/Imagenes/meme_deportes1.jpg",
        "proyect 1/Imagenes/meme_deportes2.jpg",
        "proyect 1/Imagenes/meme_deportes3.png"]

informacion = ["Comandos: (/comandos /ayuda): enseña todos los comandos",
"Hola: /hola: Bot saluda",
"Xd: /xd (numero de veces que quieras que se repita xd): Bot dice xd varias veces",
"Repetir: /repetir (lo que se va ha repetir) (las veces que se repetira): repetira lo que tu quieres la veces que quieras",
"Suma: /suma (numero) (numero): suma dos numeros",
"Escribir: /scribir (lo que quieras que escriba): el bot escribe/copia lo que le digas",
"Contraseña personalizad: /contrasena_personalizada (la contraseña que quieras) (el numero de digitos que quieras que se le añadan): se le añaden la cantidad de numeros que tu quieras a la contraseña que quieras",
"Contraseña aleatoria: /contrasena_aleatoria (los digitos que quieras que tenga la contraseña): crea una contraseña con la cantidad de digitos que quieras",
"Duplicar: /duplicar (lo que quieras que duplique): duplica lo que quieras",
"Meme1: /meme1: enseña el el meme 1",
"Meme2: /meme2: enseña el el meme 2",
"Meme aleatorio: /meme_aleatorio: enseña un meme aleatorio",
"Pato: /pato: enseña un gif de un pato",
"Meme tema: /meme_tema (el tema del que quieras que sea el meme: deportes, videojuegos, programacion)"]

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='/', intents=intents)

@bot.event
async def preparado():
    print(f'We have logged in as {bot.user}')

@bot.command()
async def comandos(ctx):
    print(ctx)
    for i in range(len(informacion)):
        await ctx.send(informacion[i])

@bot.command()
async def ayuda(ctx):
    for i in range(len(informacion)):
        await ctx.send(informacion[i])

@bot.command()
async def hola(ctx):
    await ctx.send(f'Hola, soy un bot {bot.user}!')

@bot.command()
async def xd(ctx, count_xd = 5):
    await ctx.send("xd" * count_xd)

@bot.command()
async def repetir(ctx, content = "repeating...", times = 3):
    """Repeats a message multiple times."""
    for i in range(times):
        await ctx.send(content)

@bot.command()
async def suma(ctx, left: int, right: int):
    """Adds two numbers together."""
    await ctx.send(left + right)

@bot.command()
async def escribir(ctx, *choices: str):
    """Chooses between multiple choices."""
    await ctx.send(random.choice(choices))

@bot.command()
async def contrasena_personalizada(ctx, contrasena: str, longitud: int):
    caracteres = "+-/*!&$#?=@abcdefghijklnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
    for i in range(longitud):
        x = random.randint(0, 70)
        contrasena = contrasena + caracteres[x]
    await ctx.send(contrasena)

@bot.command()
async def contrasena_aleatoria(ctx, longitud: int):
    contrasena = "Contraseña: "
    caracteres = "+-/*!&$#?=@abcdefghijklnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
    for i in range(longitud):
        contrasena = contrasena + caracteres[random.randint(1, 70)]
    await ctx.send(contrasena)

@bot.command()
async def duplicar(ctx, letter: str, str = 1):
    result = ''
    for i in range (str):
        result += letter * 2
    await ctx.send(result)

@bot.command()
async def meme1(ctx):
    with open('proyect 1/Imagenes/meme_programacion.jpeg', 'rb') as f:
        # ¡Vamos a almacenar el archivo de la biblioteca Discord convertido en esta variable!
        picture = discord.File(f)
    # A continuación, podemos enviar este archivo como parámetro.
    await ctx.send(file=picture)

@bot.command()
async def meme2(ctx):
    with open('proyect 1/Imagenes/meme2.jpeg', 'rb') as f:
        # ¡Vamos a almacenar el archivo de la biblioteca Discord convertido en esta variable!
        picture = discord.File(f)
    # A continuación, podemos enviar este archivo como parámetro.
    await ctx.send(file=picture)

@bot.command()
async def meme_aleatorio(ctx):
    imagen = random.choice(memes)
    with open(imagen, 'rb') as f:
        # ¡Vamos a almacenar el archivo de la biblioteca Discord convertido en esta variable!
        picture = discord.File(f)
    # A continuación, podemos enviar este archivo como parámetro.
    await ctx.send(file=picture)

def get_duck_image_url():    
    url = 'https://random-d.uk/api/random'
    res = requests.get(url)
    data = res.json()
    return data['url']

@bot.command('pato')
async def pato(ctx):
    '''Una vez que llamamos al comando duck, 
    el programa llama a la función get_duck_image_url'''
    image_url = get_duck_image_url()
    await ctx.send(image_url)

programacion = ["proyect 1/Imagenes/meme_programacion.jpeg", "proyect 1/Imagenes/meme2.jpeg"]
bideojuegos = ["proyect 1/Imagenes/meme_bideojuegos1.jpg", "proyect 1/Imagenes/meme_bideojuegos2.jpg"]
deportes = ["proyect 1/Imagenes/meme_deportes1.jpg", "proyect 1/Imagenes/meme_deportes2.jpg", "proyect 1/Imagenes/meme_deportes3.png"]

@bot.command()
async def meme_tema(ctx, tema: str):
    if tema == "programacion":
        imagen = random.choice(programacion)
    elif tema == "videojuegos":
        imagen = random.choice(bideojuegos)
    elif tema == "deportes":
        x = random.randint(1,5)
        if x == 1:
            imagen = "proyect 1/Imagenes/meme_deportes1.jpg"
        elif x == 2:
            imagen = "proyect 1/Imagenes/meme_deportes2.jpg"
        elif x > 2 and x < 6:
            imagen = "proyect 1/Imagenes/meme_deportes3.png"
    with open(imagen, 'rb') as f:
        picture = discord.File(f)
    await ctx.send(file=picture)

@bot.command()
async def automobilistica(ctx):
    if ctx.message.attachments:
        for attachments in ctx.message.attachments:
            file_name = attachments.filename
            image_path = f"./image/{file_name}"
            file_url = attachments.url
            await attachments.save(f"./image/{file_name}")
            result, final_result = detect_ia_on_road(model1, label1, image_path)
            for i in range(len(result)):
                await ctx.send(result[i])

            await ctx.send("Predicción final: ")
            await ctx.send(final_result[0])
            await ctx.send(final_result[1])

    else:
        await ctx.send("Olvidaste adjuntar la imagen :(")

@bot.command()
async def tecnologia(ctx):
    if ctx.message.attachments:
        for attachments in ctx.message.attachments:
            file_name = attachments.filename
            image_path = f"./image/{file_name}"
            file_url = attachments.url
            await attachments.save(f"./image/{file_name}")
            result, final_result = deteccion_tecnologica(model2, label2, image_path)
            for i in range(len(result)):
                await ctx.send(result[i])

            await ctx.send("Predicción final: ")
            await ctx.send(final_result[0])
            await ctx.send(final_result[1])

    else:
        await ctx.send("Olvidaste adjuntar la imagen :(")


bot.run("TOKEN")
