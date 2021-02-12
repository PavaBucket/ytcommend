import youtube_dl
import json


# Youtube-dl options to only list metadata
ydl_opts = {
    'quiet': True,
    'skip_download': True,
    'forceid': True,
    'forcetitle': True,
    'forceurl': True,
    'forcejson': True,
    'ignoreerrors': True,
    'download': False,
}

# Query filters
queries = ["machine+learning", "data+science", "kaggle"]

# Collect the data using youtube dl and store on json
for query in queries:
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        infoSearched = ydl.extract_info("ytsearch1000:{}".format(query))
        with open("./data/ytRawLinks.json", "+a") as output:
            for entry in infoSearched['entries']:
                if entry is not None:
                    data = {"title": entry.get('title'), "description": entry.get('description'), "upload_date": entry.get('upload_date'), "uploader": entry.get('uploader'),
                            "uploader_id": entry.get('uploader_id'), "uploader_url": entry.get('uploader_url'), "channel_id": entry.get('channel_id'), "channel_url": entry.get('channel_url'),
                            "duration": entry.get('duration'), "view_count": entry.get('view_count'), "average_rating": entry.get('average_rating'), "webpage_url": entry.get('webpage_url'),
                            "is_live": entry.get('is_live'), "like_count": entry.get('like_count'), "dislike_count": entry.get('dislike_count'), "channel": entry.get('channel'),
                            "extractor": entry.get('extractor'), "n_entries": entry.get('n_entries'), "playlist": entry.get('playlist'), "playlist_id": entry.get('playlist_id'),
                            "thumbnail": entry.get('thumbnail'), "fps": entry.get('fps'), "height": entry.get('height'),
                            "url": entry.get('url'), "width": entry.get('width'), "query": query}
                    output.write("{}\n".format(json.dumps(data)))
