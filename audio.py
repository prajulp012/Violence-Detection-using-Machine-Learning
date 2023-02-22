import moviepy
import moviepy.editor
video = moviepy.editor.VideoFileClip(r"C:\Users\praju\OneDrive\Desktop\Violence\a.mp4")
audio=video.audio
audio.write_audiofile('new_audio1.mp3')
