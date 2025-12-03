import { NextResponse } from 'next/server'
import { readFileSync, existsSync } from 'fs'
import { join } from 'path'

export async function GET() {
  try {
    const candidatesPath = join(process.cwd(), 'public', 'candidates.json')
    const validatedPath = join(process.cwd(), 'public', 'validated.json')

    if (!existsSync(candidatesPath)) {
      return NextResponse.json({ error: 'candidates.json not found' }, { status: 404 })
    }

    const candidates = JSON.parse(readFileSync(candidatesPath, 'utf8'))

    let validated: Record<string, string> = {}
    if (existsSync(validatedPath)) {
      validated = JSON.parse(readFileSync(validatedPath, 'utf8'))
    }

    return NextResponse.json({ candidates, validated })
  } catch (error) {
    return NextResponse.json({ error: 'Failed to load candidates' }, { status: 500 })
  }
}
