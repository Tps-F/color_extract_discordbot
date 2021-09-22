from PIL import Image, ImageDraw, ImageFont
import cv2
from sklearn.cluster import KMeans
import numpy as np
from numpy import linalg as LA
import random
import discord
import requests
import io

client = discord.Client()
Token = 'Enter Your TOKEN'

@client.event
async def on_message(message):
    if message.content.startswith('/pic'):
        print(message.attachments[0].url)
        r = requests.get(message.attachments[0].url)
        img = Image.open(io.BytesIO(r.content))
        #img_resize = img.resize((500), int(img.height * 500 / img.width))
        img.save("image.png")
        await message.channel.send("Please wait a moment.")
        color_arr = extract_main_color(img_path, 7)
        show_tiled_main_color(color_arr)
        #draw_random_stripe(color_arr, img_path)
        file = discord.File("./image/stripe_image.png", filename="stripe.png")
        await message.channel.send(file=file)

def download_img(url, file_name):
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(file_name, 'wb') as f:
            f.write(r.content)

def draw_random_stripe(color_arr, img_path):

    width = 1024
    height = 1024

    stripe_color_img = Image.new(
        mode='RGB', size=(width, height), color='#333333')
    current_height = 0
    while current_height < height:
        random_index = random.randrange(color_arr.shape[0])
        color_hex_str = '#%02x%02x%02x' % tuple(color_arr[random_index])
        random_height = random.randrange(5, 70)
        color_img = Image.new(
            mode='RGB', size=(width, random_height),
            color=color_hex_str)
        stripe_color_img.paste(
            im=color_img,
            box=(0, current_height))
        current_height += random_height
    stripe_color_img.show()
    #stripe_color_img.save('./image/stripe_' + img_path)


def show_tiled_main_color(color_arr):

    IMG_SIZE = 64
    MARGIN = 15
    width = IMG_SIZE * color_arr.shape[0] + MARGIN * 2
    height = IMG_SIZE + MARGIN * 2

    tiled_color_img = Image.new(
        mode='RGB', size=(width, height), color='#333333')

    for i, rgb_arr in enumerate(color_arr):
        color_hex_str = '#%02x%02x%02x' % tuple(rgb_arr)
        color_img = Image.new(
            mode='RGB', size=(IMG_SIZE, IMG_SIZE),
            color=color_hex_str)
        tiled_color_img.paste(
            im=color_img,
            box=(MARGIN + IMG_SIZE * i, MARGIN))

    tiled_color_img.show()
    tiled_color_img.save('./image/stripe_' + img_path)

def extract_main_color(img_path, color_num):

    cv2_img = cv2.imread(img_path)
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    cv2_img = cv2_img.reshape(
        (cv2_img.shape[0] * cv2_img.shape[1], 3))

    cluster = KMeans(n_clusters=color_num)
    cluster.fit(X=cv2_img)
    cluster_centers_arr = cluster.cluster_centers_.astype(
        int, copy=False)
    trans_color = cv2_img[0]
    cluster_centers_arr = np.array([i for i in cluster_centers_arr if LA.norm(np.array(i - trans_color), 2) > 50])
    print("extracted colors array:")
ff    print(cluster_centers_arr)
    return cluster_centers_arr



img_path = 'image.png'






client.run(TOKEN)