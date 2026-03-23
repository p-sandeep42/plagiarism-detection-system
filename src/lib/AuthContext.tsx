"use client";

import React, { createContext, useContext, useState, useEffect, useCallback } from "react";

interface User {
  id: number;
  email: string;
}

interface AuthContextType {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType>({
  user: null,
  token: null,
  isAuthenticated: false,
  isLoading: true,
  login: async () => {},
  register: async () => {},
  logout: () => {},
});

export function useAuth() {
  return useContext(AuthContext);
}

const API_BASE = "/api/py";

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Validate existing token on mount
  useEffect(() => {
    const stored = localStorage.getItem("auradiff_token");
    if (stored) {
      setToken(stored);
      fetch(`${API_BASE}/auth/me`, {
        headers: { Authorization: `Bearer ${stored}` },
      })
        .then((r) => {
          if (r.ok) return r.json();
          throw new Error("Invalid token");
        })
        .then((data) => setUser({ id: data.id, email: data.email }))
        .catch(() => {
          localStorage.removeItem("auradiff_token");
          setToken(null);
        })
        .finally(() => setIsLoading(false));
    } else {
      setIsLoading(false);
    }
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    const res = await fetch(`${API_BASE}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });

    if (!res.ok) {
      const body = await res.json().catch(() => null);
      throw new Error(body?.detail || "Login failed");
    }

    const data = await res.json();
    localStorage.setItem("auradiff_token", data.token);
    setToken(data.token);
    setUser({ id: 0, email: data.email });
  }, []);

  const register = useCallback(async (email: string, password: string) => {
    const res = await fetch(`${API_BASE}/auth/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });

    if (!res.ok) {
      const body = await res.json().catch(() => null);
      throw new Error(body?.detail || "Registration failed");
    }

    const data = await res.json();
    localStorage.setItem("auradiff_token", data.token);
    setToken(data.token);
    setUser({ id: 0, email: data.email });
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem("auradiff_token");
    setToken(null);
    setUser(null);
  }, []);

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        isAuthenticated: !!token,
        isLoading,
        login,
        register,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}
