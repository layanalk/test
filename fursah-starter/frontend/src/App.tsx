import { useEffect, useRef, useState } from 'react';
import type { Constraints, DiscoveryResponse, Profile, Recommendation, SimilarAlternative, ViewState } from './types';
import { discover } from './api/client';
import { loadInitialProfile, persistProfile } from './profile/profile';
import Header from './components/Header';
import { ProfileEditor, ProfileSummary } from './profile/ProfilePanel';
import SearchForm from './opportunities/SearchForm';
import SearchResults from './opportunities/SearchResults';
import OpportunityDetails from './opportunities/OpportunityDetails';

export default function App() {
  const [initial] = useState(loadInitialProfile);
  const [profile, setProfile] = useState<Profile>(initial.profile);
  const [constraints, setConstraints] = useState<Constraints>(initial.constraints);
  const [query, setQuery] = useState('');
  const [view, setView] = useState<ViewState>('initial');
  const [res, setRes] = useState<DiscoveryResponse | null>(null);
  const [editing, setEditing] = useState(false);
  const [details, setDetails] = useState<Recommendation | SimilarAlternative | null>(null);
  const queryRef = useRef<HTMLTextAreaElement>(null);
  const seq = useRef(0);
  const pending = useRef(false);
  const controller = useRef<AbortController | null>(null);
  const batchRetry = useRef<{ token: string; index: number } | null>(null);
  const submitted = useRef<{ query: string; profile: Profile; constraints: Constraints; constraint_resolution?: 'profile' | 'query' } | null>(null);
  useEffect(() => () => { ++seq.current; controller.current?.abort(); }, []);

  useEffect(() => persistProfile(profile, constraints), [profile, constraints]);

  const missing = view === 'needs_info' ? res?.missing_fields ?? [] : [];

  function invalidateSearch() {
    ++seq.current;
    controller.current?.abort();
    pending.current = false;
    submitted.current = null;
    batchRetry.current = null;
    setRes(null);
    setDetails(null);
    setView('initial');
  }

  async function runDiscover(p: Profile = profile, c: Constraints = constraints, resolution?: 'profile' | 'query', page?: { token: string; index: number }) {
    if (pending.current) return;
    const snapshot = resolution || page ? submitted.current : { query: query.trim(), profile: p, constraints: c };
    if (!snapshot) return;
    const q = snapshot.query;
    if (!q) { queryRef.current?.focus(); return; }
    // Results and resolution actions share the exact submitted input snapshot.
    submitted.current = structuredClone({ ...snapshot, constraint_resolution: resolution ?? snapshot.constraint_resolution });
    batchRetry.current = page ?? null;
    const id = ++seq.current;
    controller.current?.abort();
    controller.current = new AbortController();
    pending.current = true;
    setView('loading');
    setRes(null);
    try {
      const r = await discover({ request_id: null, ...submitted.current,
        batch_token: page?.token, batch_index: page?.index ?? 0 }, controller.current.signal);
      if (id !== seq.current) return;
      if (r.status !== 'error' || r.error_code === 'FURSAH_BATCH_EXPIRED') batchRetry.current = null;
      setRes(r);
      setView(r.status);
    } catch {
      if (id !== seq.current) return;
      setRes({ request_id: '', status: 'error', message: 'تعذّر إكمال البحث. لم تفقد بيانات ملفك.', recommendations: [], missing_fields: [], error_code: 'FURSAH_CLIENT_ERROR' });
      setView('error');
    } finally {
      if (id === seq.current) pending.current = false;
    }
  }

  return (
    <div className="app" dir="rtl" lang="ar">
      <Header onEditProfile={() => setEditing(true)} />
      <main className="container main">
        {initial.notice && <p className="notice" role="status">{initial.notice}</p>}
        <ProfileSummary profile={profile} constraints={constraints} missing={missing} onEdit={() => setEditing(true)} />
        <SearchForm
          inputRef={queryRef}
          query={query}
          onChange={value => { invalidateSearch(); setQuery(value); }}
          onSubmit={() => runDiscover()}
          loading={view === 'loading'}
        />
        <SearchResults
          view={view}
          res={res}
          onEditProfile={() => setEditing(true)}
          onResolve={choice => runDiscover(profile, constraints, choice)}
          onRetry={() => runDiscover(profile, constraints, undefined, batchRetry.current ?? undefined)}
          onFocusQuery={() => queryRef.current?.focus()}
          onOpenDetails={setDetails}
          onBatch={index => { if (res?.batch_token) runDiscover(profile, constraints, undefined, { token: res.batch_token, index }); }}
        />
      </main>

      {editing && (
        <ProfileEditor
          profile={profile}
          constraints={constraints}
          missing={missing}
          onClose={() => setEditing(false)}
          onSave={(p, c, andSearch) => {
            invalidateSearch();
            setProfile(p);
            setConstraints(c);
            setEditing(false);
            if (andSearch) runDiscover(p, c);
          }}
        />
      )}
      {details && <OpportunityDetails rec={details} onClose={() => setDetails(null)} />}
    </div>
  );
}
