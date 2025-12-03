import { NextRequest, NextResponse } from 'next/server'
import { readFileSync, writeFileSync, existsSync } from 'fs'
import { join } from 'path'

export async function POST(request: NextRequest) {
  try {
    const { videoId, decision } = await request.json()

    if (!videoId || !decision) {
      return NextResponse.json({ error: 'Missing videoId or decision' }, { status: 400 })
    }

    const validatedPath = join(process.cwd(), 'public', 'validated.json')

    let validated: Record<string, string> = {}
    if (existsSync(validatedPath)) {
      validated = JSON.parse(readFileSync(validatedPath, 'utf8'))
    }

    validated[videoId] = decision

    writeFileSync(validatedPath, JSON.stringify(validated, null, 2))

    return NextResponse.json({ success: true, total: Object.keys(validated).length })
  } catch (error) {
    return NextResponse.json({ error: 'Failed to save validation' }, { status: 500 })
  }
}
