import { NextRequest, NextResponse } from 'next/server'
import { readFile, writeFile, mkdir } from 'fs/promises'
import { join, dirname } from 'path'
import { existsSync } from 'fs'

const LABELS_FILE = join(process.cwd(), '..', 'labels', 'shooter_labels.json')

export async function POST(request: NextRequest) {
  try {
    const { videoId, label } = await request.json()

    if (!videoId || !label) {
      return NextResponse.json(
        { error: 'Missing required fields: videoId, label' },
        { status: 400 }
      )
    }

    // Ensure labels directory exists
    const labelsDir = dirname(LABELS_FILE)
    if (!existsSync(labelsDir)) {
      await mkdir(labelsDir, { recursive: true })
    }

    // Load existing labels
    let labels: Record<string, any> = {}
    if (existsSync(LABELS_FILE)) {
      try {
        const content = await readFile(LABELS_FILE, 'utf-8')
        labels = JSON.parse(content)
      } catch (e) {
        console.error('Error reading labels file:', e)
      }
    }

    // Update label
    labels[videoId] = {
      ...label,
      timestamp: new Date().toISOString(),
    }

    // Save labels
    await writeFile(LABELS_FILE, JSON.stringify(labels, null, 2))

    return NextResponse.json({
      success: true,
      totalLabeled: Object.keys(labels).filter(k => labels[k].status === 'labeled').length,
      totalSkipped: Object.keys(labels).filter(k => labels[k].status === 'skipped').length,
    })
  } catch (error) {
    console.error('Error saving label:', error)
    return NextResponse.json(
      { error: 'Failed to save label', details: String(error) },
      { status: 500 }
    )
  }
}
