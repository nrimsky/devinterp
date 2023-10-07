import os
from datetime import datetime


def run_movie_cmd(suffix=""):
    if not os.path.exists("movies"):
        os.mkdir("movies")
    mp4_name = os.path.join(
        "movies", f'movie_{datetime.now().strftime("%Y%m%d_%H%M%S")}{suffix}.mp4'
    )
    os.system(
        f"ffmpeg -framerate 3 -i frames/embeddings_movie_%06d.png -c:v libx264 -pix_fmt yuv420p {mp4_name}"
    )
    os.system("rm frames/embeddings_movie_*.png")
    print(f"movie saved as {mp4_name}")
