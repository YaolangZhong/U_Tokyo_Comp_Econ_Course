// Let the official Supabase SDK handle token expiry, refresh, and rotation.
// This adapter only chooses whether its stored session survives browser closure.
export function sessionStorageAdapter(local, tab, prefix = 'utokyo-course') {
  const preferenceKey = `${prefix}-remember`;
  let remember = tab.getItem(preferenceKey) !== 'false';
  const keys = new Set();
  const storage = {
    getItem(key) {
      keys.add(key);
      return tab.getItem(key) ?? local.getItem(key);
    },
    setItem(key, value) {
      keys.add(key);
      (remember ? local : tab).setItem(key, value);
      (remember ? tab : local).removeItem(key);
    },
    removeItem(key) {
      keys.add(key);
      local.removeItem(key);
      tab.removeItem(key);
    },
  };
  return {
    storage,
    get remember() { return remember; },
    setRemember(value) {
      const cached = [...keys].map(key => [key, storage.getItem(key)]);
      remember = Boolean(value);
      tab.setItem(preferenceKey, String(remember));
      for (const [key, data] of cached) if (data !== null) storage.setItem(key, data);
    },
    clear() {
      for (const key of keys) storage.removeItem(key);
      tab.removeItem(preferenceKey);
    },
  };
}
