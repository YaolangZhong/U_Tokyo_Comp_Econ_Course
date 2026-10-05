import test from 'node:test';
import assert from 'node:assert/strict';
import { sessionStorageAdapter } from '../assets/portal-session.mjs';
import { validateConfig, validateProfile, PROGRAMMES } from '../assets/portal-api.mjs';
function memory() {
  const data = new Map();
  return { getItem: k => data.get(k) ?? null, setItem: (k,v) => data.set(k,v), removeItem: k => data.delete(k) };
}
test('remembered session survives browser restart and token rotation', () => {
  const local = memory();
  const first = sessionStorageAdapter(local, memory());
  assert.equal(first.remember, true);
  first.storage.setItem('auth', 'initial');
  first.storage.setItem('auth', 'refreshed');
  const reopened = sessionStorageAdapter(local, memory());
  assert.equal(reopened.storage.getItem('auth'), 'refreshed');
  reopened.clear();
  assert.equal(local.getItem('auth'), null);
});
test('shared device session survives reload but not a new tab session', () => {
  const local = memory(), tab = memory();
  const first = sessionStorageAdapter(local, tab);
  first.setRemember(false);
  first.storage.setItem('auth', 'session');
  assert.equal(local.getItem('auth'), null);
  const reload = sessionStorageAdapter(local, tab);
  assert.equal(reload.remember, false);
  assert.equal(reload.storage.getItem('auth'), 'session');
  assert.equal(sessionStorageAdapter(local, memory()).storage.getItem('auth'), null);
});
test('changing preference migrates existing tokens and clear removes both stores', () => {
  const local = memory(), tab = memory(), adapter = sessionStorageAdapter(local, tab);
  adapter.storage.setItem('auth', 'session');
  adapter.setRemember(false);
  assert.equal(local.getItem('auth'), null);
  assert.equal(tab.getItem('auth'), 'session');
  adapter.setRemember(true);
  assert.equal(tab.getItem('auth'), null);
  assert.equal(local.getItem('auth'), 'session');
  adapter.clear();
  assert.equal(adapter.storage.getItem('auth'), null);
});
test('accepts only supported programme years and complete names', () => {
  for (const programme of PROGRAMMES) assert.deepEqual(validateProfile(' A ', ' B ', programme), {first_name:'A',last_name:'B',programme});
  assert.throws(() => validateProfile('', 'B', 'master_1'));
  assert.throws(() => validateProfile('A', 'B', 'phd_4'));
});
test('disabled configuration needs no credentials; enabled config rejects privileged keys', () => {
  assert.equal(validateConfig({enabled:false}), false);
  const config = { enabled:true, apiUrl:'https://example.supabase.co', publicKey:'sb_publishable_example' };
  assert.equal(validateConfig(config), true);
  for (const key of ['sb_secret_example', 'invalid', 'x.' + btoa(JSON.stringify({role:'service_role'})) + '.x']) {
    assert.throws(() => validateConfig({...config,publicKey:key}));
  }
  assert.throws(() => validateConfig({...config,apiUrl:'http://example.com'}));
});
