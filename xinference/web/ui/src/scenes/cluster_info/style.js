import { createTheme, tableCellClasses } from '@mui/material'
import Paper from '@mui/material/Paper'
import TableCell from '@mui/material/TableCell'
import TableRow from '@mui/material/TableRow'
import { styled } from '@mui/system'

export const theme = createTheme()

export const StyledTableCell = styled(TableCell)(() => ({
  [`&.${tableCellClasses.body}`]: {
    fontSize: 14,
  },
}))

export const StyledTableRow = styled(TableRow)(({ theme }) => ({
  '&:nth-of-type(odd)': {
    backgroundColor: theme.palette.action.hover,
  },
  // hide last border
  '&:last-child td, &:last-child th': {
    border: 0,
  },
}))

export const StyledPaper = styled(Paper)({
  padding: theme.spacing(2),
  display: 'flex',
  overflow: 'auto',
  flexDirection: 'column',
})
