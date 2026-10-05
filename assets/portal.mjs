import { validateConfig, validateProfile } from './portal-api.mjs';
import { sessionStorageAdapter } from './portal-session.mjs';

const el = id => document.getElementById(id);
const status = message => { el('portal-status').textContent = message; };
let client, storageChoice, currentUser, pendingEmail, expiryTimer;
const field = id => el(id).value;
function reset() {
  currentUser = null;
  clearTimeout(expiryTimer);
  el('email-form').hidden = false;
  el('code-form').hidden = true;
  el('profile-fields').disabled = true;
  el('profile-form').reset();
  el('sign-out').hidden = true;
  el('email-fields').disabled = false;
}
function authMessage(error) {
  return error?.status === 429 ? 'Please wait before requesting another code.'
    : 'Unable to complete sign-in. Check your code and connection, then try again.';
}
async function loadProfile() {
  const { data, error } = await client.auth.getUser();
  if (error || !data.user) throw error || new Error('No verified session');
  currentUser = data.user;
  if (!currentUser.email_confirmed_at) throw new Error('Email is not verified');
  const result = await client.from('registrations').select('first_name,last_name,programme')
    .eq('user_id', currentUser.id).maybeSingle();
  if (result.error) throw result.error;
  el('email-form').hidden = true;
  el('code-form').hidden = true;
  el('profile-fields').disabled = false;
  el('sign-out').hidden = false;
  el('verified-email').value = currentUser.email;
  if (result.data) {
    el('first-name').value = result.data.first_name;
    el('last-name').value = result.data.last_name;
    el('programme').value = result.data.programme;
  }
  status(result.data ? 'Welcome back. Your registration is saved.' : 'Email verified. Complete your registration below.');
}
async function initialize() {
  const config = await fetch(new URL('./portal-config.json', import.meta.url), { cache: 'no-store' }).then(r => {
    if (!r.ok) throw new Error('Configuration unavailable');
    return r.json();
  });
  if (!validateConfig(config)) return;
  // Loaded only when the registration backend has been explicitly enabled.
  const { createClient } = await import('https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2.117.2/+esm');
  storageChoice = sessionStorageAdapter(window.localStorage, window.sessionStorage);
  el('remember-device').checked = storageChoice.remember;
  client = createClient(config.apiUrl, config.publicKey, {
    auth: { persistSession: true, autoRefreshToken: true, detectSessionInUrl: false,
      storage: storageChoice.storage, storageKey: 'utokyo-course-auth-v1' },
  });
  client.auth.onAuthStateChange(event => {
    // Keep auth callbacks synchronous; avoid SDK lock deadlocks.
    if (event === 'SIGNED_OUT') {
      reset();
      status('Signed out. Enter your email when you want to sign in again.');
    }
  });
  const { data, error } = await client.auth.getSession();
  if (error) throw error;
  if (data.session) await loadProfile();
  else { reset(); status('Enter your email to receive a one-time sign-in code.'); }
}

el('email-form').addEventListener('submit', async event => {
  event.preventDefault();
  if (!client || !event.target.reportValidity()) return;
  el('email-fields').disabled = true;
  storageChoice.setRemember(el('remember-device').checked);
  pendingEmail = field('email').trim();
  try {
    const { error } = await client.auth.signInWithOtp({ email: pendingEmail, options: { shouldCreateUser: true } });
    if (error) throw error;
    el('code-form').hidden = false;
    el('code-recipient').textContent = `Enter the code sent to ${pendingEmail}.`;
    status('Check your email for a sign-in code. You can request another after a short wait.');
    el('code').focus();
    clearTimeout(expiryTimer);
    expiryTimer = setTimeout(() => { el('email-fields').disabled = false; }, 60000);
  } catch (error) { status(authMessage(error)); el('email-fields').disabled = false; }
});
el('change-email').addEventListener('click', () => {
  clearTimeout(expiryTimer);
  el('code-form').reset();
  el('code-form').hidden = true;
  el('email-fields').disabled = false;
  el('email').focus();
});
el('code-form').addEventListener('submit', async event => {
  event.preventDefault();
  if (!client || !event.target.reportValidity()) return;
  el('code-fields').disabled = true;
  try {
    const { error } = await client.auth.verifyOtp({ email: pendingEmail, token: field('code').trim(), type: 'email' });
    if (error) throw error;
    el('code').value = '';
    await loadProfile();
  } catch (error) { status(authMessage(error)); }
  finally { el('code-fields').disabled = false; }
});
el('profile-form').addEventListener('submit', async event => {
  event.preventDefault();
  if (!client || !currentUser || !event.target.reportValidity()) return;
  el('profile-fields').disabled = true;
  try {
    const profile = validateProfile(field('first-name'), field('last-name'), field('programme'));
    // The database must enforce ownership independently through row-level security.
    const { error } = await client.from('registrations').upsert({ user_id: currentUser.id, ...profile }, { onConflict: 'user_id' });
    if (error) throw error;
    status('Your registration has been saved.');
  } catch { status('Registration could not be saved. Check your details and connection, then try again.'); }
  finally { el('profile-fields').disabled = !currentUser; }
});
el('sign-out').addEventListener('click', async () => {
  el('sign-out').disabled = true;
  try {
    const { error } = await client.auth.signOut({ scope: 'local' });
    if (error) throw error;
    storageChoice.clear();
    reset();
    status('Signed out of this browser.');
  } catch {
    status('Could not finish signing out. Check your connection and retry before leaving a shared computer.');
  } finally { el('sign-out').disabled = false; }
});
initialize().catch(() => status('Registration is temporarily unavailable. Please try again later.'));
