import json
import youtube_dl
import settings


def collect():

    # Collect the data using youtube dl and store on json
    for filter in settings.queryFilters:
        with youtube_dl.YoutubeDL(settings.ydl_opts) as ydl:
            infoSearched = ydl.extract_info("{}:{}".format(settings.baseQuery, filter))
            with open(settings.rawLinksPath, "+a") as output:
                for entry in infoSearched['entries']:
                    if entry is not None:
                        data = {"title": entry.get('title'), "description": entry.get('description'), "upload_date": entry.get('upload_date'), "uploader": entry.get('uploader'),
                                "uploader_id": entry.get('uploader_id'), "uploader_url": entry.get('uploader_url'), "channel_id": entry.get('channel_id'), "channel_url": entry.get('channel_url'),
                                "duration": entry.get('duration'), "view_count": entry.get('view_count'), "average_rating": entry.get('average_rating'), "webpage_url": entry.get('webpage_url'),
                                "is_live": entry.get('is_live'), "like_count": entry.get('like_count'), "dislike_count": entry.get('dislike_count'), "channel": entry.get('channel'),
                                "extractor": entry.get('extractor'), "n_entries": entry.get('n_entries'), "playlist": entry.get('playlist'), "playlist_id": entry.get('playlist_id'),
                                "thumbnail": entry.get('thumbnail'), "fps": entry.get('fps'), "height": entry.get('height'),
                                "url": entry.get('url'), "width": entry.get('width'), "query": filter}
                        output.write("{}\n".format(json.dumps(data)))
