-- 0003: buckets privados de mídia (Storage)
-- Acesso sempre via backend (service key): upload por signed URL e download
-- por signed URL de curta duração — por isso não há policies em storage.objects.
-- Listas de MIME cobrem as variantes reais dos navegadores (ex.: audio/x-m4a
-- é como o Windows reporta .m4a).

insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values
  ('medical-videos', 'medical-videos', false, 1073741824, array[
     'video/mp4','video/quicktime','video/x-msvideo','video/webm','video/x-matroska']),
  ('medical-audio', 'medical-audio', false, 209715200, array[
     'audio/wav','audio/x-wav','audio/wave','audio/vnd.wave',
     'audio/mpeg','audio/mp3','audio/mp4','audio/x-m4a','audio/m4a','audio/aac',
     'audio/ogg','audio/webm','audio/flac']),
  ('medical-docs', 'medical-docs', false, 52428800, array[
     'application/pdf','text/plain','text/markdown',
     'image/png','image/jpeg','image/dicom','application/dicom'])
on conflict (id) do update
  set file_size_limit = excluded.file_size_limit,
      allowed_mime_types = excluded.allowed_mime_types;
