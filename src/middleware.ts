import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

// Routes that require authentication
const PROTECTED_ROUTES = ["/compare", "/batch", "/history"];

// Routes that are always public
const PUBLIC_ROUTES = ["/", "/login"];

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  // Skip API routes, static files, and Next.js internals
  if (
    pathname.startsWith("/api/") ||
    pathname.startsWith("/_next/") ||
    pathname.startsWith("/favicon") ||
    pathname.includes(".")
  ) {
    return NextResponse.next();
  }

  // Check if route is protected
  const isProtected = PROTECTED_ROUTES.some(
    (route) => pathname === route || pathname.startsWith(route + "/")
  );

  if (!isProtected) return NextResponse.next();

  // Check for auth token in cookie or header
  // Since we store in localStorage (client-side), the middleware checks
  // a cookie that the client sets. If not set, redirect to login.
  const token = request.cookies.get("auradiff_token")?.value;

  // For client-side auth (localStorage), we can't check the token server-side.
  // Instead, we rely on client-side redirects in the pages themselves.
  // The middleware here adds security headers.
  
  const response = NextResponse.next();
  
  // Security headers
  response.headers.set("X-Content-Type-Options", "nosniff");
  response.headers.set("X-Frame-Options", "DENY");
  response.headers.set("X-XSS-Protection", "1; mode=block");
  response.headers.set("Referrer-Policy", "strict-origin-when-cross-origin");
  
  return response;
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
