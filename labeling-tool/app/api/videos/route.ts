import { NextResponse } from 'next/server'
import { readdir, readFile, stat } from 'fs/promises'
import { join } from 'path'
import { existsSync } from 'fs'

const DATA_DIR = join(process.cwd(), '..', 'data', 'Basketball_51 dataset')
const PKL_DIR = join(process.cwd(), '..', 'data', 'sam3_extracted', 'sam3_extracted')
const LABELS_FILE = join(process.cwd(), '..', 'labels', 'shooter_labels.json')

interface VideoInfo {
  id: string
  path: string
  label: number
}

export async function GET() {
  try {
    const videos: VideoInfo[] = []

    // Check if PKL directory exists (has video info and labels)
    const pklDirs = ['train', 'val', 'test']

    for (const split of pklDirs) {
      const splitDir = join(PKL_DIR, split)
      if (!existsSync(splitDir)) continue

      try {
        const files = await readdir(splitDir)
        const pklFiles = files.filter(f => f.endsWith('.pkl'))

        for (const pklFile of pklFiles) {
          const id = pklFile.replace('.pkl', '')

          // Parse video path from id: ft0_v108_002649 -> ft0/ft0_v108_002649_x264.mp4
          const parts = id.split('_')
          const prefix = parts[0] // ft0, ft1, mp0, mp1
          const videoPath = join(DATA_DIR, prefix, `${id}_x264.mp4`)

          // Check if video exists
          if (existsSync(videoPath)) {
            // Determine label from prefix (ft = free throw, mp = missed?)
            // ft0 = miss (0), ft1 = make (1), mp0 = miss, mp1 = make
            const label = prefix.endsWith('1') ? 1 : 0

            videos.push({
              id,
              path: videoPath,
              label,
            })
          }
        }
      } catch (e) {
        console.error(`Error reading ${splitDir}:`, e)
      }
    }

    // Load existing labels
    let labels = {}
    if (existsSync(LABELS_FILE)) {
      try {
        const content = await readFile(LABELS_FILE, 'utf-8')
        labels = JSON.parse(content)
      } catch (e) {
        console.error('Error reading labels:', e)
      }
    }

    // Sort videos by id
    videos.sort((a, b) => a.id.localeCompare(b.id))

    return NextResponse.json({
      videos,
      labels,
      dataDir: DATA_DIR,
      pklDir: PKL_DIR,
    })
  } catch (error) {
    console.error('Error listing videos:', error)
    return NextResponse.json(
      { error: 'Failed to list videos', details: String(error) },
      { status: 500 }
    )
  }
}
