'use client'

import { useState, useEffect, useRef, useCallback } from 'react'

interface Candidate {
  video_id: string
  frame_filename: string
  release_frame_idx: number
  release_status: string
  label: number
  ball_pos: [number, number]
  ball_area: number
}

export default function ValidationTool() {
  const [candidates, setCandidates] = useState<Candidate[]>([])
  const [validated, setValidated] = useState<Record<string, string>>({})
  const [currentIndex, setCurrentIndex] = useState(0)
  const [loading, setLoading] = useState(true)
  const [imageLoaded, setImageLoaded] = useState(false)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const current = candidates[currentIndex]
  const approved = Object.values(validated).filter(v => v === 'approved').length
  const rejected = Object.values(validated).filter(v => v === 'rejected').length
  const total = candidates.length

  useEffect(() => {
    async function load() {
      const res = await fetch('/api/candidates')
      const data = await res.json()
      setCandidates(data.candidates || [])
      setValidated(data.validated || {})
      const first = (data.candidates || []).findIndex(
        (c: Candidate) => !data.validated?.[c.video_id]
      )
      if (first >= 0) setCurrentIndex(first)
      setLoading(false)
    }
    load()
  }, [])

  const drawImage = useCallback(() => {
    if (!current || !canvasRef.current) return
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const img = new Image()
    img.onload = () => {
      canvas.width = img.width
      canvas.height = img.height
      ctx.drawImage(img, 0, 0)

      if (current.ball_pos) {
        const [x, y] = current.ball_pos
        ctx.beginPath()
        ctx.arc(x, y, 18, 0, Math.PI * 2)
        ctx.strokeStyle = '#fb923c'
        ctx.lineWidth = 3
        ctx.stroke()
      }
      setImageLoaded(true)
    }
    img.src = `/frames/${current.frame_filename}`
  }, [current])

  useEffect(() => {
    setImageLoaded(false)
    drawImage()
  }, [drawImage])

  const validate = async (decision: 'approved' | 'rejected') => {
    if (!current) return
    await fetch('/api/validate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ videoId: current.video_id, decision }),
    })
    setValidated(prev => ({ ...prev, [current.video_id]: decision }))
    goNext()
  }

  const goNext = useCallback(() => {
    let next = currentIndex + 1
    while (next < candidates.length && validated[candidates[next]?.video_id]) {
      next++
    }
    if (next < candidates.length) {
      setCurrentIndex(next)
    } else if (currentIndex < candidates.length - 1) {
      setCurrentIndex(currentIndex + 1)
    }
  }, [currentIndex, candidates, validated])

  const goPrev = useCallback(() => {
    setCurrentIndex(Math.max(0, currentIndex - 1))
  }, [currentIndex])

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'a' || e.key === 'A' || e.key === 'Enter') {
        e.preventDefault()
        validate('approved')
      } else if (e.key === 'r' || e.key === 'R' || e.key === 'Backspace') {
        e.preventDefault()
        validate('rejected')
      } else if (e.key === 'ArrowLeft') {
        goPrev()
      } else if (e.key === 'ArrowRight') {
        goNext()
      }
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [current, goNext, goPrev])

  if (loading) {
    return (
      <div className="h-screen flex items-center justify-center bg-stone-50">
        <p className="text-xl tracking-[0.2em] text-stone-400 uppercase">Loading</p>
      </div>
    )
  }

  const currentStatus = current ? validated[current.video_id] : undefined

  return (
    <div className="h-screen flex flex-col bg-stone-50 overflow-hidden">

      {/* Header */}
      <header className="pt-10 pb-8 px-16 flex items-end justify-between">

        <div className="space-y-2">
          <p className="text-xs tracking-[0.25em] text-stone-500 uppercase font-medium">
            Frame Validation
          </p>
          <h1 className="text-6xl font-light tracking-tight text-stone-800">
            {currentIndex + 1}
            <span className="text-stone-300 mx-4">/</span>
            <span className="text-stone-400">{total}</span>
          </h1>
        </div>

        <div className="text-right space-y-3">
          <p className="text-sm tracking-wide text-stone-600">
            Verify court angle, ball visibility, release moment
          </p>
          <div className="flex items-center justify-end gap-12">
            <div>
              <p className="text-xs tracking-widest text-stone-500 uppercase mb-1">Approved</p>
              <p className="text-4xl font-light text-emerald-600 tracking-tight">{approved}</p>
            </div>
            <div>
              <p className="text-xs tracking-widest text-stone-500 uppercase mb-1">Rejected</p>
              <p className="text-4xl font-light text-rose-500 tracking-tight">{rejected}</p>
            </div>
          </div>
        </div>

      </header>

      {/* Canvas Area */}
      <main className="flex-1 px-16 pb-8 min-h-0">
        <div className="h-full relative rounded-xl overflow-hidden bg-black shadow-2xl">
          <canvas
            ref={canvasRef}
            className="w-full h-full object-contain"
            style={{ maxWidth: '100%', maxHeight: '100%' }}
          />

          {!imageLoaded && (
            <div className="absolute inset-0 flex items-center justify-center bg-stone-100">
              <p className="text-lg tracking-widest text-stone-400 uppercase animate-pulse">Loading</p>
            </div>
          )}

          {currentStatus && (
            <div className={`absolute top-8 right-8 px-5 py-2.5 text-sm font-medium tracking-widest uppercase ${currentStatus === 'approved' ? 'bg-emerald-500 text-white' : 'bg-rose-500 text-white'}`}>
              {currentStatus}
            </div>
          )}

          {current && (
            <div className="absolute bottom-8 left-8">
              <p className="text-xs tracking-wider text-white/60 font-mono">
                {current.video_id}
              </p>
            </div>
          )}
        </div>
      </main>

      {/* Bottom Controls */}
      <footer className="px-16 pb-12 pt-6">
        <div className="flex gap-6">
          <button
            onClick={() => validate('rejected')}
            className="flex-1 py-5 text-sm font-medium tracking-[0.2em] uppercase
              bg-white text-stone-600
              hover:bg-rose-500 hover:text-white
              transition-all duration-300 ease-out
              border border-stone-200 hover:border-rose-500"
          >
            Reject
          </button>
          <button
            onClick={() => validate('approved')}
            className="flex-1 py-5 text-sm font-medium tracking-[0.2em] uppercase
              bg-white text-stone-600
              hover:bg-emerald-500 hover:text-white
              transition-all duration-300 ease-out
              border border-stone-200 hover:border-emerald-500"
          >
            Approve
          </button>
        </div>

        <p className="text-center mt-8 text-xs tracking-widest text-stone-400 uppercase">
          A to approve  &nbsp;&nbsp;&nbsp;  R to reject  &nbsp;&nbsp;&nbsp;  Arrows to navigate
        </p>
      </footer>

    </div>
  )
}