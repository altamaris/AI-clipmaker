# Установка необходимых библиотек
!pip install opencv-python moviepy transformers openai-whisper openai python-dotenv pydub
!apt install ffmpeg

import cv2
import numpy as np
import os
from datetime import datetime, timedelta
import random
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, AudioFileClip
from pydub import AudioSegment
from IPython.display import HTML, display
from base64 import b64encode
from dotenv import load_dotenv
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

class VideoProcessor:
    """Основной класс для обработки видео"""
    def __init__(self):
        self.scene_threshold = 25  # Порог для смены сцены
        self.volume_threshold = 0.15  # Порог громкости звука
        self.min_highlight_duration = 3000  # 3 секунды минимальная длительность клипа
        self.max_highlights = 5  # Максимальное количество клипов

        # Инициализация моделей
        try:
            import whisper
            self.whisper_model = whisper.load_model("small")
            logger.info("Модель Whisper загружена")
        except Exception as e:
            logger.warning(f"Ошибка загрузки Whisper: {e}")
            self.whisper_model = None

    def process_video(self, video_path):
        """Основной метод обработки видео"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Файл {video_path} не найден")

        # 1. Анализ видео
        logger.info("Анализ видео...")
        moments = self._analyze_video(video_path)

        if not moments:
            logger.warning("Не найдено интересных моментов")
            return None

        # 2. Создание клипов
        logger.info("Создание клипов...")
        clips_info = []
        base_name = os.path.splitext(os.path.basename(video_path))[0]

        for i, moment in enumerate(moments[:self.max_highlights]):
            try:
                clip_path = f"{base_name}_highlight_{i+1}.mp4"
                self._create_highlight_clip(video_path, moment, clip_path)

                # Генерация описания
                description, hashtags = self._generate_content(
                    f"Интересный момент на {moment/1000:.1f} секунде"
                )

                clips_info.append({
                    'path': clip_path,
                    'description': description,
                    'hashtags': hashtags,
                    'time': f"{moment/1000:.1f}с"
                })

                logger.info(f"Создан клип {i+1}/{min(len(moments), self.max_highlights)}")

            except Exception as e:
                logger.error(f"Ошибка создания клипа {i+1}: {e}")
                continue

        return clips_info

    def _analyze_video(self, video_path):
        """Анализ видео для поиска интересных моментов"""
        # 1. Детекция смены сцен
        scene_changes = self._detect_scene_changes(video_path)

        # 2. Детекция громких звуков
        loud_moments = self._detect_audio_peaks(video_path)

        # Объединение и фильтрация моментов
        all_moments = sorted(list(set(scene_changes + loud_moments)))
        return self._filter_moments(all_moments)

    def _detect_scene_changes(self, video_path):
        """Обнаружение резких смен сцен"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError("Не удалось открыть видеофайл")

        detector = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=self.scene_threshold,
            detectShadows=False
        )

        changes = []
        prev_mean = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fgmask = detector.apply(gray)
            current_mean = np.mean(fgmask)

            if abs(current_mean - prev_mean) > self.scene_threshold:
                changes.append(current_ms)

            prev_mean = current_mean

        cap.release()
        return changes

    def _detect_audio_peaks(self, video_path):
        """Обнаружение громких звуков в аудио"""
        try:
            # Извлечение аудио с помощью moviepy
            audio_clip = AudioFileClip(video_path)
            audio_path = "temp_audio.wav"
            audio_clip.write_audiofile(audio_path, fps=44100)
            audio_clip.close()

            # Анализ аудио с помощью pydub
            audio = AudioSegment.from_wav(audio_path)
            os.remove(audio_path)

            chunk_size = 100  # Анализировать каждые 100 мс
            loud_moments = []

            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i+chunk_size]
                if chunk.dBFS > -20:  # Порог громкости
                    loud_moments.append(i)

            return [t for t in loud_moments if t > 0]  # Исключаем нулевой момент

        except Exception as e:
            logger.error(f"Ошибка анализа аудио: {e}")
            return []

    def _filter_moments(self, moments):
        """Фильтрация и объединение близких моментов"""
        if not moments:
            return []

        filtered = [moments[0]]
        for moment in moments[1:]:
            if moment - filtered[-1] > self.min_highlight_duration:
                filtered.append(moment)

        return filtered

    def _create_highlight_clip(self, video_path, moment_ms, output_path):
        """Создание клипа с субтитрами"""
        # Оптимальная длительность для Shorts/Reels (15-60 секунд)
        start = max(0, moment_ms - 7000)  # 7 сек до момента
        end = min(
            moment_ms + 7000,  # 7 сек после момента
            VideoFileClip(video_path).duration * 1000  # Не превышаем длительность видео
        )

        # Корректировка длительности
        if end - start > 60000:  # Максимум 60 секунд
            end = start + 60000
        elif end - start < 15000:  # Минимум 15 секунд
            end = start + 15000
            if end > VideoFileClip(video_path).duration * 1000:
                start = max(0, end - 15000)

        with VideoFileClip(video_path) as video:
            clip = video.subclip(start/1000, end/1000)

            # Генерация и добавление субтитров
            subtitled_clip = self._add_subtitles(clip)

            # Оптимизированное сохранение
            subtitled_clip.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                fps=24,
                threads=4,
                preset='fast',
                bitrate='5000k'
            )

    def _add_subtitles(self, clip):
        """Добавление субтитров с помощью Whisper"""
        if not self.whisper_model:
            logger.warning("Whisper не загружен - субтитры не будут добавлены")
            return clip

        try:
            audio_path = "temp_audio.wav"
            clip.audio.write_audiofile(audio_path, fps=16000)  # Whisper работает лучше на 16kHz

            result = self.whisper_model.transcribe(audio_path)
            os.remove(audio_path)

            text_clips = []
            for segment in result['segments']:
                txt_clip = TextClip(
                    segment['text'],
                    fontsize=28,
                    color='white',
                    bg_color='rgba(0,0,0,0.7)',
                    font='Arial-Bold',
                    size=(clip.w*0.9, None),
                    method='caption'
                ).set_start(segment['start']).set_duration(segment['end']-segment['start'])
                txt_clip = txt_clip.set_position(('center', 'bottom'))
                text_clips.append(txt_clip)

            return CompositeVideoClip([clip] + text_clips)

        except Exception as e:
            logger.error(f"Ошибка генерации субтитров: {e}")
            return clip  # Возвращаем клип без субтитров в случае ошибки

    def _generate_content(self, context):
        """Генерация текста и хэштегов"""
        # Резервные варианты
        default_desc = f"Интересный момент: {context}"
        default_tags = ["#видео", "#контент", "#интересно", "#познавательно", "#топ"]

        return default_desc, default_tags

class ContentPlanner:
    """Класс для планирования публикаций"""
    def __init__(self):
        self.platform_schedule = {
            'tiktok': {
                'best_times': ['07:00', '10:00', '22:00'],
                'max_posts_per_day': 3
            },
            'instagram': {
                'best_times': ['09:00', '12:00', '19:00'],
                'max_posts_per_day': 2
            },
            'youtube': {
                'best_times': ['12:00', '16:00', '20:00'],
                'max_posts_per_day': 1
            }
        }

    def generate_schedule(self, clips_data, platform='tiktok'):
        """Генерация расписания публикаций"""
        if platform not in self.platform_schedule:
            raise ValueError(f"Платформа {platform} не поддерживается")

        config = self.platform_schedule[platform]
        schedule = []
        current_date = datetime.now()

        for i, clip in enumerate(clips_data):
            # Распределяем публикации по дням
            day_offset = i // config['max_posts_per_day']
            post_date = current_date + timedelta(days=day_offset)

            # Выбираем случайное оптимальное время
            post_time = random.choice(config['best_times'])

            schedule.append({
                'date': post_date.strftime(f"%Y-%m-%d {post_time}"),
                'platform': platform,
                'clip_path': clip['path'],
                'description': clip['description'],
                'hashtags': " ".join(clip['hashtags']),
                'time_in_video': clip['time']
            })

        return schedule

def upload_video():
    """Функция для загрузки видео в Colab"""
    from google.colab import files
    uploaded = files.upload()
    for filename in uploaded.keys():
        logger.info(f"Загружено видео: {filename}")
        return filename
    return None

def show_video(video_path):
    """Функция для отображения видео в Colab"""
    try:
        mp4 = open(video_path,'rb').read()
        data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
        return HTML(f"""
        <video width=400 controls>
            <source src="{data_url}" type="video/mp4">
        </video>
        """)
    except Exception as e:
        logger.error(f"Ошибка отображения видео: {e}")
        return HTML("<p>Не удалось загрузить видео</p>")

def main():
    print("=== Умный монтажер видео для соцсетей ===")
    print("Автоматическое создание Shorts/Reels из длинных видео\n")

    # 1. Загрузка видео
    print("Шаг 1: Загрузите видеофайл (MP4, MOV)...")
    video_path = upload_video()
    if not video_path:
        print("Ошибка: видео не загружено")
        return

    # 2. Обработка видео
    print("\nШаг 2: Обработка видео...")
    processor = VideoProcessor()
    clips_data = processor.process_video(video_path)

    if not clips_data:
        print("Не удалось создать клипы")
        return

    # 3. Показ результатов
    print("\nСозданные клипы:")
    for i, clip in enumerate(clips_data, 1):
        print(f"\nКлип #{i} (момент {clip['time']}):")
        print(f"Описание: {clip['description']}")
        print(f"Хэштеги: {' '.join(clip['hashtags'])}")
        display(show_video(clip['path']))

    # 4. Планирование публикаций
    print("\nШаг 3: Планирование публикаций...")
    planner = ContentPlanner()

    print("\nВарианты публикации в TikTok:")
    tiktok_plan = planner.generate_schedule(clips_data, platform='tiktok')
    for post in tiktok_plan:
        print(f"\nДата: {post['date']}")
        print(f"Описание: {post['description']}")
        print(f"Хэштеги: {post['hashtags']}")

    print("\nГотово! Все клипы сохранены в текущей директории.")

if __name__ == "__main__":
    main()
