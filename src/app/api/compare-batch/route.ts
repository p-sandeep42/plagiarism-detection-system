import { NextRequest, NextResponse } from 'next/server';

export async function POST(req: NextRequest) {
  const formData = await req.formData();
  const fastapiUrl = process.env.NEXT_PUBLIC_FASTAPI_URL;

  try {
    const upstream = await fetch(`${fastapiUrl}/api/py/compare-batch`, {
      method: 'POST',
      body: formData,
    });

    if (!upstream.ok) {
      const err = await upstream.text();
      return NextResponse.json({ error: err }, { status: upstream.status });
    }

    const data = await upstream.json();
    return NextResponse.json(data);
  } catch (e) {
    console.error('API Error:', e);
    return NextResponse.json({ error: 'Backend unreachable' }, { status: 502 });
  }
}
