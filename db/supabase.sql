-- 잡핏 에이전트 — Supabase pgvector 스키마.
-- Supabase SQL Editor에서 한 번 실행한다. 임베딩 차원(1536)은
-- app/vectorstore.py의 EMBEDDING_MODEL(text-embedding-3-small)과 맞춘다.

create extension if not exists vector;

create table if not exists profile_chunks (
  id text primary key,           -- chunk-001 ... (근거 인용 단위)
  content text not null,
  embedding vector(1536) not null
);

-- 정책 없이 RLS만 켠다: anon 키로는 접근 불가, 서버의 service_role 키만 접근한다.
alter table profile_chunks enable row level security;

create or replace function match_profile_chunks(
  query_embedding vector(1536),
  match_count int default 5
)
returns table (id text, content text, similarity float)
language sql stable
as $$
  select id, content, 1 - (embedding <=> query_embedding) as similarity
  from profile_chunks
  order by embedding <=> query_embedding
  limit match_count;
$$;
