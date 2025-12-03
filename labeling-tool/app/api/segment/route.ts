import { NextRequest, NextResponse } from 'next/server'
import Replicate from 'replicate'
import { join } from 'path'
import { existsSync, readFileSync } from 'fs'

const CACHE_DIR = join(process.cwd(), 'public', 'frames')

export async function POST(request: NextRequest) {
  try {
    const { videoId, clickX, clickY } = await request.json()

    if (!videoId || clickX === undefined || clickY === undefined) {
      return NextResponse.json(
        { error: 'Missing required fields: videoId, clickX, clickY' },
        { status: 400 }
      )
    }

    // Check for API token
    const apiToken = process.env.REPLICATE_API_TOKEN
    if (!apiToken) {
      return NextResponse.json(
        { error: 'REPLICATE_API_TOKEN not configured. Add it to .env.local' },
        { status: 500 }
      )
    }

    // Get frame image as base64
    const frameFile = join(CACHE_DIR, `${videoId}.jpg`)
    if (!existsSync(frameFile)) {
      return NextResponse.json(
        { error: `Frame not found: ${videoId}` },
        { status: 404 }
      )
    }

    const imageBuffer = readFileSync(frameFile)
    const base64Image = `data:image/jpeg;base64,${imageBuffer.toString('base64')}`

    // Call Replicate SAM 2
    const replicate = new Replicate({
      auth: apiToken,
    })

    const output = await replicate.run(
      'meta/sam-2-image:fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83',
      {
        input: {
          image: base64Image,
          point_coords: `${clickX},${clickY}`,
          point_labels: '1',
          multimask_output: false,
        },
      }
    )

    // Output should be a mask image URL or base64
    if (output && typeof output === 'object') {
      // Replicate returns an object with combined_mask or similar
      const maskUrl = (output as any).combined_mask || (output as any).masks?.[0] || output

      if (typeof maskUrl === 'string' && maskUrl.startsWith('http')) {
        // Fetch the mask and convert to base64 for display
        const maskResponse = await fetch(maskUrl)
        const maskBuffer = await maskResponse.arrayBuffer()
        const maskBase64 = `data:image/png;base64,${Buffer.from(maskBuffer).toString('base64')}`
        return NextResponse.json({ mask: maskBase64 })
      } else if (typeof maskUrl === 'string') {
        return NextResponse.json({ mask: maskUrl })
      }
    }

    // Fallback: return raw output
    return NextResponse.json({ mask: null, raw: output })
  } catch (error) {
    console.error('Segmentation error:', error)
    return NextResponse.json(
      { error: 'Segmentation failed', details: String(error) },
      { status: 500 }
    )
  }
}
