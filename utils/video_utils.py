import os


def collect_videos():
    videos = {}
    VIDEOS_DIR = './videos/input_videos'
    for video in os.listdir(VIDEOS_DIR):
        video_path = os.path.join(VIDEOS_DIR, video)
        video = video.replace('.mp4', '')
        videos.setdefault(video, video_path)

    return videos
