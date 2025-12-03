import { NextRequest, NextResponse } from 'next/server'
import { exec } from 'child_process'
import { promisify } from 'util'
import { join } from 'path'
import { existsSync, mkdirSync, readFileSync } from 'fs'

const execAsync = promisify(exec)

const DATA_DIR = join(process.cwd(), '..', 'data', 'Basketball_51 dataset')
const CACHE_DIR = join(process.cwd(), 'public', 'frames')

// Ensure cache directory exists
if (!existsSync(CACHE_DIR)) {
  mkdirSync(CACHE_DIR, { recursive: true })
}

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const { id } = params
    const frameFile = join(CACHE_DIR, `${id}.jpg`)

    // Check cache first
    if (existsSync(frameFile)) {
      const buffer = readFileSync(frameFile)
      return new NextResponse(buffer, {
        headers: {
          'Content-Type': 'image/jpeg',
          'Cache-Control': 'public, max-age=86400',
        },
      })
    }

    // Parse video path from id: ft0_v108_002649 -> ft0/ft0_v108_002649_x264.mp4
    const parts = id.split('_')
    const prefix = parts[0]
    const videoPath = join(DATA_DIR, prefix, `${id}_x264.mp4`)

    if (!existsSync(videoPath)) {
      return NextResponse.json(
        { error: `Video not found: ${videoPath}` },
        { status: 404 }
      )
    }

    // Extract frame 5 (good shooting pose) using ffmpeg
    // Frame 5 at ~30fps = ~0.17 seconds
    const frameTime = '00:00:00.17'

    try {
      await execAsync(
        `ffmpeg -ss ${frameTime} -i "${videoPath}" -vframes 1 -q:v 2 "${frameFile}" -y`,
        { timeout: 30000 }
      )
    } catch (ffmpegError) {
      // Try alternative: extract first frame
      try {
        await execAsync(
          `ffmpeg -i "${videoPath}" -vframes 1 -q:v 2 "${frameFile}" -y`,
          { timeout: 30000 }
        )
      } catch (e) {
        return NextResponse.json(
          { error: 'FFmpeg failed to extract frame', details: String(e) },
          { status: 500 }
        )
      }
    }

    if (!existsSync(frameFile)) {
      return NextResponse.json(
        { error: 'Frame extraction produced no output' },
        { status: 500 }
      )
    }

    const buffer = readFileSync(frameFile)
    return new NextResponse(buffer, {
      headers: {
        'Content-Type': 'image/jpeg',
        'Cache-Control': 'public, max-age=86400',
      },
    })
  } catch (error) {
    console.error('Error extracting frame:', error)
    return NextResponse.json(
      { error: 'Failed to extract frame', details: String(error) },
      { status: 500 }
    )
  }
}
