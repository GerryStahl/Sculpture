// mask_subject — Apple Vision subject-isolation CLI
// Usage: mask_subject <input_image_path> <output_png_path>
// Outputs a PNG with the background made transparent (white in JPG mode).

import Foundation
import Vision
import CoreImage
import CoreGraphics
import UniformTypeIdentifiers
import ImageIO

func die(_ msg: String) -> Never {
    fputs("error: \(msg)\n", stderr)
    exit(1)
}

guard CommandLine.arguments.count == 3 else {
    die("Usage: mask_subject <input> <output_png>")
}

let inputPath  = CommandLine.arguments[1]
let outputPath = CommandLine.arguments[2]
let inputURL   = URL(fileURLWithPath: inputPath)
let outputURL  = URL(fileURLWithPath: outputPath)

// ── Load source image ────────────────────────────────────────────────────────
guard let src = CIImage(contentsOf: inputURL) else {
    die("Cannot load image: \(inputPath)")
}

let ciContext = CIContext()

// ── Run VNGenerateForegroundInstanceMaskRequest ──────────────────────────────
let handler = VNImageRequestHandler(ciImage: src, options: [:])
let request = VNGenerateForegroundInstanceMaskRequest()

do {
    try handler.perform([request])
} catch {
    die("Vision request failed: \(error)")
}

guard let result = request.results?.first else {
    die("No mask result returned from Vision")
}

// ── Build per-pixel mask covering all foreground instances ───────────────────
let allInstances = result.allInstances
let maskPixelBuf: CVPixelBuffer
do {
    maskPixelBuf = try result.generateScaledMaskForImage(forInstances: allInstances, from: handler)
} catch {
    die("generateScaledMaskForImage failed: \(error)")
}

let maskCI = CIImage(cvPixelBuffer: maskPixelBuf)

// ── Composite: subject over transparent background ───────────────────────────
// Blend formula: out_alpha = mask, out_rgb = src_rgb * mask
let blended = src.applyingFilter("CIBlendWithMask", parameters: [
    kCIInputMaskImageKey: maskCI,
    kCIInputBackgroundImageKey: CIImage.empty()
])

// ── Render to PNG ────────────────────────────────────────────────────────────
guard let cgImage = ciContext.createCGImage(blended, from: blended.extent) else {
    die("Failed to render CGImage")
}

let dest = CGImageDestinationCreateWithURL(outputURL as CFURL, UTType.png.identifier as CFString, 1, nil)!
CGImageDestinationAddImage(dest, cgImage, nil)
guard CGImageDestinationFinalize(dest) else {
    die("Failed to write PNG to \(outputPath)")
}

print("saved: \(outputPath)")
