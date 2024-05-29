import logging
import argparse
from google.cloud import storage
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from tqdm import tqdm
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
import requests
from bs4 import BeautifulSoup
import re

# Import the necessary modules
from components.utils import gcs_utils  # Naming it GCS causes some issues with the imports

# Configuration
DEVELOPER_KEY = "AIzaSyD1sC-t8Z8F1FNHyp0FlhLnbilEudYrG-I"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_youtube_service():
    """
    Create a YouTube service object using the developer key.
    """
    return build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY, cache_discovery=False)


def get_youtube_id(url, id_type='channel'):
    """
    Retrieve the YouTube channel ID or playlist ID from a given URL.

    Parameters:
    url (str): The URL of the YouTube page.
    id_type (str): The type of ID to retrieve ('channel' or 'playlist').

    Returns:
    str: The requested ID.

    Raises:
    Exception: If the page fails to load or the ID is not found.
    """
    # Fetch the HTML content of the page
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception(f"Failed to load page {url}")
    
    page_content = response.content
    
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(page_content, 'html.parser')
    
    # Find the ID in the page content
    # The IDs are often located in a script tag containing JSON data
    scripts = soup.find_all('script')
    id_pattern = None
    
    if id_type == 'channel':
        id_pattern = r'"channelId":"(UC[0-9A-Za-z_-]{22})"'
    elif id_type == 'playlist':
        id_pattern = r'"playlistId":"(PL[0-9A-Za-z_-]+)"'
    else:
        raise ValueError("Invalid id_type. Must be 'channel' or 'playlist'.")
    
    requested_id = None
    
    for script in scripts:
        if id_type in script.text:
            # Use a regular expression to extract the requested ID
            match = re.search(id_pattern, script.text)
            if match:
                requested_id = match.group(1)
                break
    
    if requested_id:
        return requested_id
    else:
        raise Exception(f"{id_type.capitalize()} ID not found in the page content")


def get_playlist_videos(youtube, playlist_id, max_number_videos=None):
    """
    Retrieves videos from a YouTube playlist.

    Args:
        youtube: The YouTube service object.
        playlist_id (str): The ID of the YouTube playlist.
        max_number_videos (int, optional): The maximum number of videos to retrieve.

    Returns:
        list: A list of video items.
    """
    try:
        next_page_token = None
        videos = []

        while True:
            playlist_items_response = youtube.playlistItems().list(
                playlistId=playlist_id,
                part="snippet",
                maxResults=50,
                pageToken=next_page_token
            ).execute()

            if 'items' not in playlist_items_response:
                raise Exception("Playlist ID not found or invalid.")

            videos.extend(playlist_items_response.get("items", []))
            if max_number_videos and len(videos) >= max_number_videos:
                videos = videos[:max_number_videos]
                break
            next_page_token = playlist_items_response.get("nextPageToken")
            if not next_page_token:
                break

        return videos
    except Exception as e:
        logger.error(f"An error occurred while retrieving playlist videos: {e}")
        return []

def get_youtube_captions(video_id, language=None):
    """
    Retrieves captions for a YouTube video in a specific language.

    Args:
        video_id (str): The ID of the YouTube video.
        language (str, optional): The language code of the desired captions. 
                                  If None, all available languages will be returned.

    Returns:
        dict: The captions in the specified language, or a dictionary containing 
              transcripts in all available languages if language is None.
    """
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcripts = {}
        if language:
            for transcript in transcript_list:
                if transcript.language_code == language:
                    captions = transcript.fetch()
                    transcripts[transcript.language_code] = ' '.join(c['text'] for c in captions)
                    return transcripts
            return None  # Language not found
        else:
            for transcript in transcript_list:
                captions = transcript.fetch()
                transcripts[transcript.language_code] = ' '.join(c['text'] for c in captions)
            return transcripts
    except Exception as e:
        logger.error(f"An error occurred while retrieving captions: {e}")
        return None

def get_videos_and_captions(url, id_type='channel', language='en', max_number_videos=None):
    """
    Lists the videos from a YouTube channel or playlist, retrieving captions if available.

    Args:
        channel_or_playlist_id (str): The ID of the YouTube channel or playlist.
        is_playlist (bool): Set to True if the ID is a playlist ID, False if it's a channel ID.
        language (str, optional): The language code of the desired captions.
        max_number_videos (int, optional): The maximum number of videos to retrieve.

    Returns:
        dict: A dictionary containing video details and captions.
    """
    try:
        youtube = get_youtube_service()
        channel_or_playlist_id = get_youtube_id(url=url, id_type=id_type)

        if id_type== 'playlist':
            playlist_id = channel_or_playlist_id
        else:
            channels_response = youtube.channels().list(
                id=channel_or_playlist_id,
                part="contentDetails"
            ).execute()

            if 'items' not in channels_response:
                raise Exception("Channel ID not found or invalid.")

            uploads_list_id = channels_response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
            playlist_id = uploads_list_id

        video_items = get_playlist_videos(youtube, playlist_id, max_number_videos)
        
        if not video_items:
            return {}
        
        video_items = sorted(video_items, key=lambda x: x["snippet"]["publishedAt"], reverse=True)

        total_videos = len(video_items)
        video_data = {}
        processed_videos = 0

        with tqdm(total=total_videos, desc="Processing videos") as pbar:
            for item in video_items:
                video_id = item["snippet"]["resourceId"]["videoId"]
                try:
                    captions = get_youtube_captions(video_id, language) if language else None
                    tittle = item["snippet"]["title"]
                    print("Tittle", tittle)
                    video_data[video_id] = {
                        "title": tittle ,
                        "published_at": item["snippet"]["publishedAt"],
                        "channel_id": item["snippet"]["channelId"],
                        "description": item["snippet"]["description"],
                        "thumbnails": item["snippet"]["thumbnails"],
                        f"captions_{language}": captions.get(language) if captions else None if language else None
                    }
                except Exception as e:
                    logger.error(f"Error processing video {video_id}: {e}")

                processed_videos += 1
                pbar.update(1)

        return video_data
    except Exception as e:
        logger.error(f"An error occurred while retrieving videos: {e}")
        return {}

# Example usage

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and store YouTube video captions.")
    parser.add_argument('--url', type=str, required=True, help='The ID of the YouTube channel or playlist.')
    parser.add_argument('--blob_name', type=str, required=True, help='The name of the blob in the GCS bucket.')
    parser.add_argument('--id_type', type=str, help='The ID of the YouTube channel or playlist.')
    parser.add_argument('--bucket_name', type=str, default="metal-sky-419309-videos-v1", help='The name of the GCS bucket.')
    parser.add_argument('--language', type=str, default="en", help='The language code of the desired captions.')
    parser.add_argument('--max_number_videos', type=int, default=None, help='The maximum number of videos to retrieve.')
    args = parser.parse_args()

    videos = get_videos_and_captions(args.url, args.id_type, args.language, args.max_number_videos)
    if videos:
        gcs_utils.save_dict_to_gcs(videos, args.bucket_name, args.blob_name)
    else:
        logger.error("No video data to save.")
