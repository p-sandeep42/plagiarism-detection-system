import { NextRequest, NextResponse } from 'next/server';

export async function POST(req: NextRequest) {
  const formData = await req.formData();
  const fastapiUrl = process.env.NEXT_PUBLIC_FASTAPI_URL || 'http://localhost:8000';

  try {
    // Forward auth header
    const headers: Record<string, string> = {};
    const authHeader = req.headers.get('authorization');
    if (authHeader) {
      headers['Authorization'] = authHeader;
    }

    const upstream = await fetch(`${fastapiUrl}/compare-batch`, {
      method: 'POST',
      body: formData,
      headers,
    });

    if (!upstream.ok) {
      const err = await upstream.text();
      // Don't forward raw stack traces — return a sanitized message
      let message = 'An error occurred during batch analysis.';
      try {
        const parsed = JSON.parse(err);
        message = parsed.detail || parsed.error || message;
      } catch {}
      return NextResponse.json({ error: message }, { status: upstream.status });
    }

    const data = await upstream.json();
    return NextResponse.json(data);
  } catch (e) {
    console.error('Batch API proxy error:', e);
    return NextResponse.json(
      { error: 'Backend unreachable. Please try again later.' },
      { status: 502 }
    );
  }
}
