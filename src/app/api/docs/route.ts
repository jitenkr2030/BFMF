import { NextRequest, NextResponse } from 'next/server';
import { readFile } from 'fs/promises';
import { join } from 'path';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const doc = searchParams.get('doc') || 'index.md';
    
    // Security check to prevent directory traversal
    if (doc.includes('..') || doc.includes('/')) {
      return NextResponse.json({ error: 'Invalid document path' }, { status: 400 });
    }
    
    const filePath = join(process.cwd(), 'docs', doc);
    
    try {
      const content = await readFile(filePath, 'utf-8');
      
      // Set appropriate headers for markdown content
      return new NextResponse(content, {
        headers: {
          'Content-Type': 'text/markdown; charset=utf-8',
          'Cache-Control': 'public, max-age=3600',
        },
      });
    } catch (fileError) {
      return NextResponse.json({ error: 'Document not found' }, { status: 404 });
    }
  } catch (error) {
    console.error('Error serving documentation:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}