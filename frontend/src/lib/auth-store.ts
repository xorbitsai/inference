export type AuthState =
  | {
      type: 'anonymous';
    }
  | {
      type: 'token';
      token: string;
    }
  | null;

let authState: AuthState = null;

export const authStore = {
  get: () => authState,

  set: (state: AuthState) => {
    authState = state;
  },

  clear: () => {
    authState = null;
  },
};